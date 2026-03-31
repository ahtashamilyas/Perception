# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
import json,uuid,joblib,os,sys,argparse
from datareader import *
from estimater import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')
sys.path.append(f'{code_dir}/GroundedSAM_demo')
import yaml
from GroundedSAM_demo.grounded_sam import GroundedSAM


def get_mask_grounded_sam(reader, i_frame, ob_id, grounded_sam_model=None, prompt_text="objects"):
  """
  Get mask using GroundedSAM instead of pre-existing masks.
  
  Args:
    reader: YcbVideoReader instance
    i_frame: Frame index
    ob_id: Object ID (used for logging, but GroundedSAM will detect objects based on prompt)
    grounded_sam_model: Initialized GroundedSAM model
    prompt_text: Text prompt for GroundingDINO (e.g., "bottle", "mustard", "objects")
  
  Returns:
    valid: Boolean mask of detected object
  """
  if grounded_sam_model is None:
    raise ValueError("GroundedSAM model must be provided")
  
  # Get the color image from the reader
  color = reader.get_color(i_frame) # H x W x 3 (BGR)
  H, W = color.shape[:2]
  
  # Use GroundedSAM to generate masks
  color_rgb =  cv2.cvtColor(color, cv2.COLOR_BGR2RGB) # H x W x 3 (RGB)
  detections = grounded_sam_model.generate_masks(color_rgb)
  
  if detections is None or len(detections["masks"]) == 0:
    logging.warning(f"No objects detected with prompt '{prompt_text}' in frame {i_frame}")
    # Return empty mask
    valid = np.zeros((H, W), dtype=bool)
  else:
    # For simplicity, we'll take the first (most confident) mask
    # In a more sophisticated implementation, you could select based on object properties
    masks = detections["masks"]  # [B', H, W]
    mask_scores = detections["masks_scores"]  # [B']
    
    # Get the mask with highest confidence
    best_mask_idx = torch.argmax(mask_scores)
    best_mask = masks[best_mask_idx]  # [H, W]
    
    # Convert to numpy boolean mask
    if torch.is_tensor(best_mask):
      valid = best_mask.cpu().numpy().astype(bool)
    else:
      valid = best_mask.astype(bool)
    
    logging.info(f"Selected mask with confidence {mask_scores[best_mask_idx]:.3f} for ob_id {ob_id}")
  
  return valid


def get_mask(reader, i_frame, ob_id, detect_type, grounded_sam_model=None, prompt_text="objects"):
  """
  Updated get_mask function that can use either original methods or GroundedSAM
  """
  if detect_type == 'grounded_sam':
    return get_mask_grounded_sam(reader, i_frame, ob_id, grounded_sam_model, prompt_text)
  elif detect_type=='box':
    mask = reader.get_mask(i_frame, ob_id)
    H,W = mask.shape[:2]
    vs,us = np.where(mask>0)
    umin = us.min()
    umax = us.max()
    vmin = vs.min()
    vmax = vs.max()
    valid = np.zeros((H,W), dtype=bool)
    valid[vmin:vmax,umin:umax] = 1
  elif detect_type=='mask':
    mask = reader.get_mask(i_frame, ob_id, type='mask_visib')
    valid = mask>0
  elif detect_type=='cnos':   #https://github.com/nv-nguyen/cnos
    mask = cv2.imread(reader.color_files[i_frame].replace('rgb','mask_cnos'), -1)
    valid = mask==ob_id
  else:
    raise RuntimeError

  return valid



def run_pose_estimation_worker(reader, i_frames, est:FoundationPose, debug=False, ob_id=None, device:int=0, grounded_sam_model=None, prompt_text="objects"):
  result = NestDict()
  torch.cuda.set_device(device)
  est.to_device(f'cuda:{device}')
  est.glctx = dr.RasterizeCudaContext(device)
  debug_dir = est.debug_dir

  for i in range(len(i_frames)):
    i_frame = i_frames[i]
    id_str = reader.id_strs[i_frame]
    logging.info(f"{i}/{len(i_frames)}, video:{reader.get_video_id()}, id_str:{id_str}")
    color = reader.get_color(i_frame)
    depth = reader.get_depth(i_frame)

    H,W = color.shape[:2]
    scene_ob_ids = reader.get_instance_ids_in_image(i_frame)
    video_id = reader.get_video_id()

    logging.info(f"video:{reader.get_video_id()}, id_str:{id_str}, ob_id:{ob_id}")
    if ob_id not in scene_ob_ids:
      logging.info(f'skip {ob_id} as it does not exist in this scene')
      continue
    
    # Always use GroundedSAM for mask generation
    ob_mask = get_mask(reader, i_frame, ob_id, detect_type='grounded_sam', grounded_sam_model=grounded_sam_model, prompt_text=prompt_text)

    # est.gt_pose = reader.get_gt_pose(i_frame, ob_id)  # optional (not needed)
    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=ob_id, iteration=5)
    logging.info(f"pose:\n{pose}")
    
    
    # Visualization
    to_origin, extents = trimesh.bounds.oriented_bounds(est.mesh_ori)
    bbox = np.array([[-extents[0]/2, -extents[1]/2, -extents[2]/2],
                     [extents[0]/2, extents[1]/2, extents[2]/2]])
    center_pose = pose @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
    
    # Visualize the GroundedSAM detected mask
    mask_vis = color.copy()
    mask_vis[ob_mask] = mask_vis[ob_mask] * 0.7 + np.array([0, 255, 0]) * 0.3  # Green overlay for GroundedSAM
    
    cv2.imshow('Pose Estimation', vis[...,::-1])
    cv2.imshow('Detected Mask', mask_vis[...,::-1])
    cv2.waitKey(1)



    if debug>=3:
      tmp = est.mesh_ori.copy()
      tmp.apply_transform(pose)
      tmp.export(f'{debug_dir}/model_tf.obj')

    result[video_id][id_str][ob_id] = pose

  return result


def run_pose_estimation():
  wp.force_load(device='cuda')
  video_dirs = sorted(glob.glob(f'{opt.ycbv_dir}/test/*'))
  res = NestDict()

  debug = opt.debug
  use_reconstructed_mesh = opt.use_reconstructed_mesh
  debug_dir = opt.debug_dir

  # Initialize GroundedSAM by default
  grounded_sam_model = None
  logging.info("Initializing GroundedSAM model...")
  try:
    # Try to suppress the problematic logging issue
    import warnings
    warnings.filterwarnings("ignore")
    
    
    # Initialize GroundedSAM
    grounded_sam = GroundedSAM.load_grounded_sam_model(
        checkpoints_dir="GroundedSAM_demo/checkpoints",
        grounded_dino_config_dir="GroundedSAM_demo/cfg/gdino",
        box_threshold=0.3,
        text_threshold=0.3,
        use_yolo_sam=True,
        sam_vit_model="mobile_sam.pt",
        mask_threshold=0.01,
        prompt_text="red mug",
        segmentor_width_size=None,
        device=None
    )
    logging.info("GroundedSAM model initialized successfully")
  except AttributeError as e:
    if "info_once" in str(e):
      logging.error("GroundingDINO logging compatibility issue detected.")
      logging.info("This is a known issue with some versions of GroundingDINO.")
      logging.info("Please check that you have the correct version of groundingdino-py installed.")
      logging.info("Try: pip install groundingdino-py==0.4.0")
    else:
      logging.error(f"AttributeError initializing GroundedSAM: {e}")
    return
  except Exception as e:
    logging.error(f"Failed to initialize GroundedSAM: {e}")
    logging.info("Cannot proceed without GroundedSAM")
    return

  reader_tmp = YcbVideoReader(video_dirs[0])
  glctx = dr.RasterizeCudaContext()
  mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4))
  est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(), symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, debug_dir=debug_dir, debug=debug)

  ob_ids = reader_tmp.ob_ids

  # Process only the first object as requested
  # if len(ob_ids) > 0:
  #   ob_ids = [ob_ids[0]]
  #   logging.info(f"Processing only the first object: {ob_ids[0]}")

  for ob_id in ob_ids:
    if use_reconstructed_mesh:
      mesh = reader_tmp.get_reconstructed_mesh(ob_id, ref_view_dir=opt.ref_view_dir)
    else:
      mesh = reader_tmp.get_gt_mesh(ob_id)
    symmetry_tfs = reader_tmp.symmetry_tfs[ob_id]

    args = []
    for video_dir in video_dirs:
      reader = YcbVideoReader(video_dir, zfar=1.5)
      scene_ob_ids = reader.get_instance_ids_in_image(0)
      if ob_id not in scene_ob_ids:
        continue
      video_id = reader.get_video_id()

      for i in range(len(reader.color_files)):
        if not reader.is_keyframe(i):
          continue
        args.append((reader, [i], est, debug, ob_id, 0, grounded_sam_model, opt.prompt_text))

    est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=symmetry_tfs, mesh=mesh)
    outs = []
    for arg in args:
      out = run_pose_estimation_worker(*arg)
      outs.append(out)

    for out in outs:
      for video_id in out:
        for id_str in out[video_id]:
          res[video_id][id_str][ob_id] = out[video_id][id_str][ob_id]

  with open(f'{opt.debug_dir}/ycbv_res.yml','w') as ff:
    yaml.safe_dump(make_yaml_dumpable(res), ff)



if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--ycbv_dir', type=str, default="/home/student/Desktop/perception/FoundationPose/demo_data/YCB_Video", help="data dir")
  parser.add_argument('--use_reconstructed_mesh', type=int, default=0)
  parser.add_argument('--ref_view_dir', type=str, default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16")
  parser.add_argument('--debug', type=int, default=0)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  
  # GroundedSAM arguments
  parser.add_argument('--use_grounded_sam', action='store_true', help="Use GroundedSAM for mask generation instead of ground truth masks")
  parser.add_argument('--grounded_sam_checkpoints_dir', type=str, default=f'{code_dir}/checkpoints', help="Directory containing GroundedSAM checkpoints")
  parser.add_argument('--grounded_dino_config_dir', type=str, default=f'{code_dir}/GroundedSAM_demo/cfg/gdino', help="Directory containing GroundingDINO config files")
  parser.add_argument('--prompt_text', type=str, default="objects", help="Text prompt for GroundingDINO object detection (e.g., 'bottle', 'mustard', 'objects')")
  
  opt = parser.parse_args()
  os.environ["YCB_VIDEO_DIR"] = opt.ycbv_dir

  set_seed(0)

  detect_type = 'grounded_sam'   # Always use GroundedSAM for mask detection

  # Usage examples:
  # 1. Use GroundedSAM with generic "objects" prompt (default):
  #    python run_ycb_video_original.py
  #
  # 2. Use GroundedSAM with specific object prompt:
  #    python run_ycb_video_original.py --prompt_text "bottle"
  #    python run_ycb_video_original.py --prompt_text "mustard container"

  run_pose_estimation()