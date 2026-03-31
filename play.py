# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
import os
import cv2
import numpy as np
from ultralytics import SAM
import imageio
import yaml
from geometry_msgs.msg import Pose, PoseStamped
from scipy.spatial.transform import Rotation as R


def detect_colored_objects(img_rgb, color_name):
  """Detect objects of specific color and return bounding boxes (xyxy)."""
  hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)  # Reader returns RGB

  # Define HSV ranges for different colors
  color_ranges = {
    'red': [(np.array([0, 50, 50]), np.array([10, 255, 255])),
            (np.array([170, 50, 50]), np.array([180, 255, 255]))],  # Red wraps around
    'blue': [(np.array([105, 50, 50]), np.array([130, 255, 255]))],
    'yellow': [(np.array([20, 100, 100]), np.array([35, 255, 255]))]
  }

  if color_name not in color_ranges:
    return []

  # Create mask for the color
  mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
  for lower, upper in color_ranges[color_name]:
    mask += cv2.inRange(hsv, lower, upper)

  # Find contours and convert to bounding boxes
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  bboxes = []
  for contour in contours:
    if cv2.contourArea(contour) > 500:  # Filter small noise
      x, y, w, h = cv2.boundingRect(contour)
      bboxes.append([x, y, x + w, y + h])

  return bboxes   # xyxy


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  # parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument("--model", type=str, default="sam_b.pt", help="Colors to detect")
  parser.add_argument("--colors", nargs='+', default=['red', 'blue', 'yellow'], help="Colors to detect")
  parser.add_argument("--cam_K", type=json.loads, default="[[912.7279052734375, 0.0, 667.5955200195312], [0.0, 911.0028076171875, 360.5406799316406], [0.0, 0.0, 1.0]]", help="Camera intrinsic parameters")
  parser.add_argument("--mask_out_dir", type=str, default=f'{code_dir}/demo_data/cube/masks', help="Directory to save per-color masks")
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  # Initialize SAM model and mask variables
  seg_model = SAM(args.model)
  seg_model.to("cuda")
  os.makedirs(args.mask_out_dir, exist_ok=True)
  ob_mask = None

  for i in range(10):
    logging.info(f'i:{i}')
    color_img = reader.get_color(i)  # RGB image (H, W, 3)
    depth = reader.get_depth(i)
    cam_k = reader.K  # Use reader's camera matrix
    
    if i==0:
      all_masks = []
      mask_info = []

      # For each requested color, detect bboxes and run SAM to get instance masks
      for color_name in args.colors:
        bboxes = detect_colored_objects(color_img, color_name)
        if not bboxes:
          logging.info(f"No {color_name} objects found.")
          continue

        # Run SAM segmentation for detected boxes
        results = seg_model.predict(source=color_img, bboxes=bboxes, device="cuda")

        if results and getattr(results[0], 'masks', None) is not None:
          masks_np = results[0].masks.data.cpu().numpy()  # (N, H, W) in {0,1}

          # Save individual masks for this color and collect
          for mi, m in enumerate(masks_np):
            m_bin = (m > 0.5).astype(np.uint8) * 255
            filename = f"{color_name}_cube_{mi:02d}.png"
            filepath = os.path.join(args.mask_out_dir, filename)
            cv2.imwrite(filepath, m_bin)
            all_masks.append((m_bin > 0))
            mask_info.append(f"{color_name}_{mi}")
            logging.info(f"Saved: {filename}")

      # Combine masks and create visualization
      if all_masks:
        # Union mask for registration
        union_mask = np.any(np.stack(all_masks, axis=0), axis=0)

        # Visualization similar to provided logic
        combined_vis = np.zeros_like(all_masks[0], dtype=np.uint8)
        for idx, m_bool in enumerate(all_masks):
          combined_vis = np.maximum(combined_vis, (m_bool.astype(np.uint8) * 255) // len(all_masks) * (idx + 1))

        cv2.imwrite(os.path.join(args.mask_out_dir, "all_cubes_combined.png"), combined_vis)
        logging.info(f"Total objects detected: {len(all_masks)}; saved combined mask to {args.mask_out_dir}")

        ob_mask = union_mask.astype(np.uint8) * 255
        
        # Debug information
        print(f"Mask shape: {ob_mask.shape}, Mask sum: {ob_mask.sum()}")
        print(f"Depth shape: {depth.shape}, Depth min/max: {depth.min():.3f}/{depth.max():.3f}")
        print(f"Valid depth pixels: {(depth > 0.001).sum()}")
        print(f"Mask coverage: {(ob_mask > 0).sum()} pixels")
        print(f"Overlapping valid pixels: {((depth > 0.001) & (ob_mask > 0)).sum()}")
        
      else:
        logging.warning("No objects detected by SAM; using empty mask.")
        ob_mask = np.zeros(color_img.shape[:2], dtype=np.uint8)

      pose = est.register(K=cam_k, rgb=color_img, depth=depth, ob_mask=ob_mask, iteration=args.est_refine_iter)
      
      # Calculate center_pose by applying inverse of to_origin transform
      center_pose = pose @ np.linalg.inv(to_origin)
      
      # Extract position and rotation from center_pose
      position = center_pose[:3, 3]
      rotation_matrix = center_pose[:3, :3]
      
      # Convert rotation matrix to quaternion
      # Note: scipy's as_quat() returns [x, y, z, w] format
      quaternion = R.from_matrix(rotation_matrix).as_quat()
      
      # Print pose information
      print(f"\nPose at frame 0:")
      print(f"Position: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
      print(f"Quaternion [x,y,z,w]: [{quaternion[0]:.4f}, {quaternion[1]:.4f}, {quaternion[2]:.4f}, {quaternion[3]:.4f}]")

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color_img[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      print("min, max depth", depth.min(), depth.max())
      print('depth.sum() 2', depth.sum())
      print("min, max color", color_img.min(), color_img.max())
      print("color.sum() 2", color_img.sum())
      print("Mesh vertices sum", mesh.vertices.sum())
      if ob_mask is not None:
        print("Mask sum", ob_mask.sum())
    #   pose = est.track_one(rgb=color_img, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))
    print(f"Pose: \n{pose}")

    with open(f'/home/student/Desktop/perception/FoundationPose/debug/ycbv_res.yml','w') as ff:
      yaml.safe_dump(make_yaml_dumpable(quaternion), ff)

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color_img, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color_img, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(3000)


    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

