import argparse
import cv2
import distinctipy
import os
import json
import torch

import numpy as np

from GroundedSAM_demo.grounded_sam import GroundedSAM

from typing_extensions import Union

from estimater import FoundationPose

import trimesh

from Utils import *

BOP_DEPTH_SCALE = 1.0
ZFAR = np.inf


def get_pose(device: Union[int, str],
                              checkpoints_dir: str, grounded_dino_config_dir: str, grounded_dino_use_vitb: bool,
                              box_threshold: float,  text_threshold: float,
                              use_yolo_sam: bool, sam_vit_model: str, mask_threshold: float,
                              prompt_text: str,
                              background: str,
                              depth_path: str,
                              mesh_path: str,
                              mesh_obj_id: int,
                              intrinsics_path: str):
    print("Starting process to get the grounded segmentation masks of video frames.")

    # File extension sets (must be defined before any usage in list comprehensions)
    image_exts = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.exr'}
    video_exts = {'.mp4', '.avi', '.mkv', '.mov'}
    npy_ext = '.npy'

    # initialize grounded sam and foundation pose
    print("Initializing models.")
    grounded_sam = GroundedSAM.load_grounded_sam_model(checkpoints_dir=checkpoints_dir,
                                                       grounded_dino_config_dir=grounded_dino_config_dir,
                                                       grounded_dino_use_vitb=grounded_dino_use_vitb,
                                                       box_threshold=box_threshold,
                                                       text_threshold=text_threshold,
                                                       use_yolo_sam=use_yolo_sam,
                                                       sam_vit_model=sam_vit_model,
                                                       mask_threshold=mask_threshold,
                                                       prompt_text=prompt_text,
                                                       segmentor_width_size=None,
                                                       device=None)
    
    glctx = dr.RasterizeCudaContext()
    mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4))
    est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(), 
                         symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, 
                         debug_dir=args.debug_dir, debug=args.debug)

    # -------------------- Prepare RGB source (mirrors depth handling) --------------------
    rgb_source_type = None  # one of ['webcam','video','single_image','image_sequence']
    rgb_single_frame = None
    rgb_sequence_files = []
    # indices managed via shared frame_idx later
    camera = None
    rgb_video_cap = None

    # Interpret 'device' argument: if path exists decide type, else attempt webcam/video integer index or filename
    if isinstance(device, str) and os.path.exists(device):
        if os.path.isdir(device):
            rgb_sequence_files = sorted([
                os.path.join(device, f) for f in os.listdir(device)
                if os.path.splitext(f.lower())[1] in image_exts
            ])
            if not rgb_sequence_files:
                raise FileNotFoundError(f"No RGB image files found in directory: {device}")
            rgb_source_type = 'image_sequence'
            print(f"Loaded RGB image sequence with {len(rgb_sequence_files)} frames from {device}")
        else:
            ext_rgb = os.path.splitext(device.lower())[1]
            if ext_rgb in image_exts:
                rgb_single_frame = cv2.imread(device, cv2.IMREAD_COLOR)
                if rgb_single_frame is None:
                    raise IOError(f"Failed to read RGB image: {device}")
                rgb_source_type = 'single_image'
                print(f"Loaded single RGB image {device}")
            elif ext_rgb in video_exts:
                rgb_video_cap = cv2.VideoCapture(device)
                if not rgb_video_cap.isOpened():
                    raise IOError(f"Failed to open RGB video: {device}")
                rgb_source_type = 'video'
                print(f"Opened RGB video {device}")
            else:
                # fallback try webcam cast
                try:
                    cam_index = int(device)
                    camera = cv2.VideoCapture(cam_index)
                    if not camera.isOpened():
                        raise IOError(f"Failed to open webcam index {cam_index}")
                    rgb_source_type = 'webcam'
                    print(f"Opened webcam at index {cam_index}")
                except ValueError:
                    raise ValueError(f"Unrecognized RGB device/path: {device}")
    else:
        # treat as webcam index
        try:
            cam_index = int(device)
            camera = cv2.VideoCapture(cam_index)
            if not camera.isOpened():
                raise IOError(f"Failed to open webcam index {cam_index}")
            rgb_source_type = 'webcam'
            print(f"Opened webcam at index {cam_index}")
        except ValueError:
            # string path that doesn't exist
            raise FileNotFoundError(f"Provided device path does not exist: {device}")

    print("Initialized models and prepared RGB source.")


    # -------------------- Prepare depth source --------------------
    depth_source_type = None  # one of ['single_image','image_sequence','npy_single','npy_sequence','video']
    depth_single_frame = None
    depth_sequence_files = []
    # indices managed via shared frame_idx later
    depth_npy_array = None
    depth_video_cap = None


    if os.path.isdir(depth_path):
        # directory of image sequence
        depth_sequence_files = sorted([
            os.path.join(depth_path, f) for f in os.listdir(depth_path)
            if os.path.splitext(f.lower())[1] in image_exts
        ])
        if not depth_sequence_files:
            raise FileNotFoundError(f"No depth image files found in directory: {depth_path}")
        depth_source_type = 'image_sequence'
        print(f"Loaded depth image sequence with {len(depth_sequence_files)} frames from {depth_path}")
    else:
        ext = os.path.splitext(depth_path.lower())[1]
        if ext in image_exts:
            depth_single_frame = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_single_frame is None:
                raise IOError(f"Failed to read depth image: {depth_path}")
            depth_source_type = 'single_image'
            print(f"Loaded single depth image {depth_path}")
        elif ext == npy_ext:
            depth_npy_array = np.load(depth_path)
            if depth_npy_array.ndim == 2:
                depth_source_type = 'npy_single'
                print(f"Loaded single depth npy array with shape {depth_npy_array.shape} from {depth_path}")
            elif depth_npy_array.ndim == 3:
                depth_source_type = 'npy_sequence'
                print(f"Loaded depth npy sequence with {depth_npy_array.shape[0]} frames from {depth_path}")
            else:
                raise ValueError(f"Unsupported depth npy shape {depth_npy_array.shape}; expected (H,W) or (N,H,W)")
        elif ext in video_exts:
            depth_video_cap = cv2.VideoCapture(depth_path)
            if not depth_video_cap.isOpened():
                raise IOError(f"Failed to open depth video: {depth_path}")
            depth_source_type = 'video'
            print(f"Opened depth video {depth_path}")
        else:
            raise ValueError(f"Unrecognized depth_path extension: {ext}. Supported: image {image_exts}, video {video_exts}, npy {npy_ext}, or directory of images.")

    # -------------------- Load mesh (if provided) --------------------
    loaded_mesh = None
    if mesh_path is not None:
        candidate_path = mesh_path
        if os.path.isdir(mesh_path):
            # If directory and obj_id provided, try standard naming patterns
            if mesh_obj_id is not None:
                patterns = [
                    f"obj_{mesh_obj_id:06d}.ply",
                    f"obj_{mesh_obj_id:06d}.obj",
                    f"{mesh_obj_id}.ply",
                    f"{mesh_obj_id}.obj",
                ]
                for p in patterns:
                    test_path = os.path.join(mesh_path, p)
                    if os.path.exists(test_path):
                        candidate_path = test_path
                        break
            # If still directory, attempt to find first mesh file
            if os.path.isdir(candidate_path):
                for fname in os.listdir(candidate_path):
                    if fname.lower().endswith(('.ply', '.obj')):
                        candidate_path = os.path.join(candidate_path, fname)
                        break
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(f"Mesh file not found (after resolution): {candidate_path}")
        try:
            loaded_mesh = trimesh.load(candidate_path, process=False)
            # If units likely in mm (heuristic: bounding box large), optionally scale (TODO: expose flag if needed)
            if loaded_mesh.scale > 1000:  # crude heuristic; leave unchanged otherwise
                pass
            symmetry_tfs = np.array([np.eye(4)])
            est.reset_object(model_pts=loaded_mesh.vertices.copy(),
                             model_normals=loaded_mesh.vertex_normals.copy(),
                             symmetry_tfs=symmetry_tfs, mesh=loaded_mesh)
            print(f"Loaded mesh from {candidate_path} (verts={len(loaded_mesh.vertices)})")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load mesh at {candidate_path}: {e}")
    else:
        print("No mesh_path provided; using placeholder box mesh.")


    # -------------------- Synchronization setup --------------------
    def _depth_len():
        if depth_source_type == 'image_sequence':
            return len(depth_sequence_files)
        if depth_source_type == 'npy_sequence':
            return depth_npy_array.shape[0]
        return None
    def _rgb_len():
        if rgb_source_type == 'image_sequence':
            return len(rgb_sequence_files)
        return None

    d_len = _depth_len()
    r_len = _rgb_len()
    if d_len is not None and r_len is not None:
        sync_len = min(d_len, r_len)
    else:
        sync_len = d_len if r_len is None else r_len  # if one is streaming, limit by finite one

    # Load intrinsics (can return a 3x3 np.ndarray or a dict per frame)
    intrinsics_data = None
    if intrinsics_path:
        try:
            intrinsics_data, BOP_DEPTH_SCALE = load_K(intrinsics_path)
            print(f"Loaded intrinsics data {intrinsics_path}")
            
            if isinstance(intrinsics_data, dict):
                print(f"Loaded intrinsics dict with {len(intrinsics_data)} entries from {intrinsics_path}")
            elif isinstance(intrinsics_data, np.ndarray):
                print(f"Loaded intrinsics matrix from {intrinsics_path}:\n{intrinsics_data}")
        except Exception as e:
            print(f"Warning: failed to load intrinsics from {intrinsics_path}: {e}. Will use fallback.")
            intrinsics_data = None

    # expected maximal number of different objects/needed color in an image
    num_max_objs = 50
    colors = distinctipy.get_colors(num_max_objs)

    print("Start getting the grounded segmentation masks for all frames.")

    cv2.namedWindow("GroundedSam", cv2.WINDOW_NORMAL)

    frame_idx = 0
    while True:
        if sync_len is not None and frame_idx >= sync_len:
            print("Reached end of synchronized sequences.")
            break

        # -------------------- Fetch RGB frame --------------------
        if rgb_source_type == 'webcam':
            ret, original_frame = camera.read()
            if not ret:
                print("Failed to grab frame from webcam.")
                continue
        elif rgb_source_type == 'video':
            ret, original_frame = rgb_video_cap.read()
            if not ret:
                rgb_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, original_frame = rgb_video_cap.read()
                if not ret:
                    print("Failed to read frame from RGB video; ending loop.")
                    break
        elif rgb_source_type == 'single_image':
            original_frame = rgb_single_frame.copy()
        elif rgb_source_type == 'image_sequence':
            seq_path = rgb_sequence_files[frame_idx % len(rgb_sequence_files)] if sync_len is None else rgb_sequence_files[frame_idx]
            original_frame = cv2.imread(seq_path, cv2.IMREAD_COLOR)
            if original_frame is None:
                print(f"Failed to read RGB frame: {seq_path}")
                break
        else:
            print("Unknown RGB source type; aborting.")
            break

        cv2.normalize(original_frame, original_frame, 0, 255, cv2.NORM_MINMAX)

        # -------------------- Fetch depth frame --------------------
        if depth_source_type == 'single_image':
            depth_frame = depth_single_frame
        elif depth_source_type == 'image_sequence':
            d_seq_path = depth_sequence_files[frame_idx % len(depth_sequence_files)] if sync_len is None else depth_sequence_files[frame_idx]
            # depth_frame = cv2.imread(d_seq_path, cv2.IMREAD_UNCHANGED)
            depth_frame = get_depth(depth_file=d_seq_path)
        elif depth_source_type == 'npy_single':
            depth_frame = depth_npy_array
        elif depth_source_type == 'npy_sequence':
            if sync_len is None:
                depth_frame = depth_npy_array[frame_idx % depth_npy_array.shape[0]]
            else:
                depth_frame = depth_npy_array[frame_idx]
        elif depth_source_type == 'video':
            ret_d, depth_frame = depth_video_cap.read()
            if not ret_d:
                depth_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_d, depth_frame = depth_video_cap.read()
                if not ret_d:
                    print("Failed to read depth frame from video; skipping pose estimation for this frame.")
                    depth_frame = None
        else:
            depth_frame = None

        if depth_frame is not None:
            if depth_frame.ndim == 3 and depth_frame.shape[2] > 1:
                depth_frame = depth_frame[:, :, 0]
            depth = depth_frame.astype(np.float32)
        else:
            depth = None
        # depth_v = depth>=0.001
        # print('depth_v.sum() 2', depth_v.sum())

        # -------------------- Detection --------------------
        rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        detections = grounded_sam.generate_masks(rgb_frame)

        if detections:
            masks = detections["masks"].squeeze(1).cpu().numpy()
            masks_scores = detections["masks_scores"].cpu().numpy()
            sort_indicis = np.argsort(-masks_scores)
            masks = masks[sort_indicis]
            masks_scores = masks_scores[sort_indicis]

            if background == "rgb":
                frame = rgb_frame.copy()
            elif background == "gray":
                frame = cv2.cvtColor(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            elif background == "black":
                frame = np.zeros_like(rgb_frame, dtype=np.uint8)
            elif background == "white":
                frame = np.ones_like(rgb_frame, dtype=np.uint8)
            else:
                raise NotImplementedError("Value for 'background' must be one of ['rgb', 'gray', 'black', 'white']")

            alpha = 0.3
            for mask_idx, mask in enumerate(masks):
                r = int(255 * colors[mask_idx][0])
                g = int(255 * colors[mask_idx][1])
                b = int(255 * colors[mask_idx][2])
                frame[mask, 0] = alpha * r + (1 - alpha) * frame[mask, 0]
                frame[mask, 1] = alpha * g + (1 - alpha) * frame[mask, 1]
                frame[mask, 2] = alpha * b + (1 - alpha) * frame[mask, 2]
        else:
            masks = None


        # -------------------- Pose estimation --------------------
        if masks is not None and depth is not None and loaded_mesh is not None:

            most_confident_mask = masks[0]
            
            # Debug: Get mask center for comparison
            mask_center = get_mask_center_2d(most_confident_mask)
            print(f"Mask center (2D): {mask_center}")
            
            out = run_pose_estimation_worker(most_confident_mask, est, intrinsics_data, ob_id=mesh_obj_id, device=0, depth_image=depth, rgb_image=rgb_frame)
        else:
            out = None
            mask_center = None

        # -------------------- Display --------------------
        if masks is not None:
            disp_frame = frame
        else:
            disp_frame = rgb_frame
        # -------------------- Visualization overlays (using embedded function) --------------------
        if out is not None and loaded_mesh is not None:
            try:
                # Since pose estimation is inaccurate, use mask-based visualization instead
                vis_img = visualize_pose_result_with_mask_center(
                    color=disp_frame,
                    mask=most_confident_mask,
                    depth=depth,
                    K=intrinsics_data,
                    scale=8.0
                )
                    
                disp_frame = vis_img
            except Exception as e:
                print(f"Visualization error: {e}")
                import traceback
                traceback.print_exc()
                print(f"Visualization error: {e}")

        cv2.imshow("GroundedSam", cv2.cvtColor(disp_frame, cv2.COLOR_RGB2BGR))

        frame_idx += 1
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            print("Escape pressed, closing.")
            break

    # clean up
    if camera is not None:
        camera.release()
    if rgb_video_cap is not None:
        rgb_video_cap.release()
    if depth_video_cap is not None:
        depth_video_cap.release()
        cv2.destroyAllWindows()

        print("Finished processing the frames.")
        print("End")

def visualize_pose_result_with_mask_center(color, mask, depth, K, scale=5.0):
    """
    Alternative visualization that places axes at the mask center using depth
    """
    try:
        # Get mask center in 2D
        mask_center = get_mask_center_2d(mask)
        if mask_center is None:
            return color
            
        cx, cy = mask_center
        
        # Get depth at mask center
        if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
            depth_at_center = depth[cy, cx]
            if depth_at_center <= 0:
                # If no depth at center, try nearby pixels
                y_min, y_max = max(0, cy-5), min(depth.shape[0], cy+5)
                x_min, x_max = max(0, cx-5), min(depth.shape[1], cx+5)
                depth_region = depth[y_min:y_max, x_min:x_max]
                valid_depths = depth_region[depth_region > 0]
                if len(valid_depths) > 0:
                    depth_at_center = np.median(valid_depths)
                else:
                    depth_at_center = 50.0  # fallback
        else:
            depth_at_center = 50.0  # fallback
            
        print(f"Mask center: {mask_center}, Depth: {depth_at_center:.1f}cm")
        
        # Create a simple pose at the mask center
        # Convert 2D + depth to 3D
        x_3d = (cx - K[0, 2]) * depth_at_center / K[0, 0]
        y_3d = (cy - K[1, 2]) * depth_at_center / K[1, 1]
        z_3d = depth_at_center
        
        # Create identity rotation at this 3D position
        simple_pose = np.eye(4)
        simple_pose[:3, 3] = [x_3d, y_3d, z_3d]
        
        print(f"Simple pose 3D position: [{x_3d:.1f}, {y_3d:.1f}, {z_3d:.1f}]")
        
        # Draw axes at this position
        vis = draw_xyz_axis(color, ob_in_cam=simple_pose, scale=scale, K=K, thickness=5, transparency=0, is_input_rgb=True)
        
        # Also draw mask center cross for reference
        vis = draw_mask_center_cross(vis, mask_center, color=(255, 0, 255), size=10, thickness=2)
        
        return vis
        
    except Exception as e:
        print(f"Error in alternative visualization: {e}")
        return color

def visualize_pose_result(color, pose, K, mesh, debug_dir=None, frame_id=None, show_window=True):
    try:
        # Try using the original pose directly first
        vis = draw_xyz_axis(color, ob_in_cam=pose, scale=5.0, K=K, thickness=5, transparency=0, is_input_rgb=True)
        
        # Debug: Calculate where the 3D pose origin projects to in 2D
        pose_3d_origin = pose[:3, 3].reshape(1, 3)  # [x, y, z] translation
        if pose_3d_origin[0, 2] > 0:  # z > 0 (in front of camera)
            # Project 3D point to 2D image coordinates
            projected = (K @ pose_3d_origin.T).T  # K * [x, y, z]^T
            pose_2d = projected[:, :2] / projected[:, 2:3]  # [u, v] = [x/z, y/z]
            pose_2d = pose_2d.astype(int)[0]
            print(f"3D pose origin projects to 2D: {pose_2d}")
        else:
            print("3D pose behind camera")
        
        print(f"Original pose translation: {pose[:3, 3]}")
        
        if debug_dir and frame_id:
            os.makedirs(f'{debug_dir}/pose_vis', exist_ok=True)
            cv2.imwrite(f'{debug_dir}/pose_vis/{frame_id}.png', vis[...,::-1])
        
        return vis
        
    except Exception as e:
        print(f"Error in visualize_pose_result: {e}")
        import traceback
        traceback.print_exc()
        return color

def get_mask_center_2d(mask):
    """Get the 2D center of a binary mask"""
    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) == 0:
        return None
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))
    return (center_x, center_y)

def draw_mask_center_cross(image, center, color=(255, 0, 255), size=20, thickness=3):
    """Draw a cross at the mask center for debugging"""
    if center is None:
        return image
    cx, cy = center
    # Draw cross
    cv2.line(image, (cx-size, cy), (cx+size, cy), color, thickness)
    cv2.line(image, (cx, cy-size), (cx, cy+size), color, thickness)
    return image

def run_pose_estimation_worker(mask, est:FoundationPose, internal_cam_parameters, ob_id=None, device:int=0, depth_image=None, rgb_image=None):
    result = NestDict()
    torch.cuda.set_device(device)
    est.to_device(f'cuda:{device}')
    est.glctx = dr.RasterizeCudaContext(device)

    # est.gt_pose = reader.get_gt_pose(i_frame, ob_id)
    # print("mask sum:", np.sum(mask))
    # print("depth min/max:", np.min(depth_image), np.max(depth_image))
    # print("depth sum:", np.sum(depth_image))
    # print("Camera intrinsics:", internal_cam_parameters)
    pose = est.register(K=internal_cam_parameters, rgb=rgb_image, depth=depth_image, ob_mask=mask)
    logging.info(f"pose:\n{pose}")

    result = pose

    return result


def load_K(path="/home/student/Desktop/perception/FoundationPose/demo_data/YCB_Video/test/000048/scene_camera.json"):


    K_table = {}
    with open(path,'r') as ff:
        info = json.load(ff)
    for k in info:
        K_table[f'{int(k):06d}'] = np.array(info[k]['cam_K']).reshape(3,3)
        bop_depth_scale = info[k]['depth_scale']

    K = list(K_table.values())[0]
    return K, bop_depth_scale


def draw_axes(image, K, pose, axis_length=5, thickness=3):
    """Draw 3D coordinate axes of an object pose onto an RGB image (in-place).

    Args:
        image (np.ndarray): RGB image (H,W,3) in RGB ordering.
        K (np.ndarray): 3x3 intrinsic matrix.
        pose (np.ndarray): 4x4 object pose (object in camera coordinates).
        axis_length (float): Length of axes in object units (e.g., meters).
        thickness (int): Line thickness.
    """
    if pose.shape != (4,4):
        return
    axes_obj = np.array([[0,0,0],
                         [axis_length,0,0],
                         [0,axis_length,0],
                         [0,0,axis_length]], dtype=np.float32)
    homog = np.concatenate([axes_obj, np.ones((4,1), dtype=np.float32)], axis=1)
    cam_pts = (pose @ homog.T).T[:, :3]
    zs = cam_pts[:,2]
    valid = zs > 1e-6
    if not np.all(valid):
        cam_pts = cam_pts[valid]
    if cam_pts.shape[0] < 4:
        return
    uv = (K @ cam_pts.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    uv = uv.astype(int)
    o,x,y,z = uv[0],uv[1],uv[2],uv[3]
    h,w = image.shape[:2]
    def inb(pt): return 0 <= pt[0] < w and 0 <= pt[1] < h
    if inb(o) and inb(x):
        cv2.line(image, tuple(o), tuple(x), (255,0,0), thickness)
    if inb(o) and inb(y):
        cv2.line(image, tuple(o), tuple(y), (0,255,0), thickness)
    if inb(o) and inb(z):
        cv2.line(image, tuple(o), tuple(z), (0,0,255), thickness)

def project_mesh_overlay(image, K, pose, est:FoundationPose, color=(0,255,255)):
    """Project mesh vertices to image and draw small points."""
    if est.mesh is None:
        return
    verts = np.asarray(est.mesh.vertices)
    if verts.shape[0] == 0:
        return
    step = max(1, verts.shape[0] // 1500)
    verts = verts[::step]
    homog = np.concatenate([verts, np.ones((verts.shape[0],1), dtype=np.float32)], axis=1)
    cam_pts = (pose @ homog.T).T[:, :3]
    mask = cam_pts[:,2] > 1e-6
    cam_pts = cam_pts[mask]
    if cam_pts.shape[0] == 0:
        return
    proj = (K @ cam_pts.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    h,w = image.shape[:2]
    for pt in proj:
        u,v = int(pt[0]), int(pt[1])
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(image, (u,v), 1, color, -1)

def get_depth(filled=False, depth_file=None):
    if filled:
        depth = cv2.imread(depth_file,-1)/1e3
    else:
        depth = cv2.imread(depth_file,-1)*1e-3*BOP_DEPTH_SCALE
    # if self.resize!=1:
    #     depth = cv2.resize(depth, fx=self.resize, fy=self.resize, dsize=None, interpolation=cv2.INTER_NEAREST)
    #     depth[depth<0.001] = 0
    #     depth[depth>ZFAR] = 0
    return depth

    # (Removed stray call to self.make_id_strs which was invalid in this context)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Applies 'GroundedSam' to all video frames.")

    # G_SAM args
    parser.add_argument("-i", "--device", dest="device", type=str, default=0,
                        help="Device/Video file for the video camera.")
    
    parser.add_argument("--depth_path", type=str, help="Path to the depth data (single image, directory of images, .npy, or video).")
    parser.add_argument("--intrinsics_path", type=str, default="/home/student/Desktop/perception/FoundationPose/demo_data/YCB_Video/test/000048/scene_camera.json", help="Path to camera intrinsics file (scene_camera.json, JSON, YAML, or txt).")
    parser.add_argument("--mesh_path", type=str, default=None, help="Path to mesh file (.ply/.obj) or directory containing mesh files.")
    parser.add_argument("--mesh_obj_id", type=int, default=None, help="Object id used to resolve mesh filename inside mesh directory (e.g., obj_000001.ply).")

    parser.add_argument("-e", "--checkpoints_dir", dest="checkpoints_dir", type=str, default="GroundedSAM_demo/checkpoints",
                    help="Path of the root dir containing the checkpoints.")
    parser.add_argument("-d", "--grounded_dino_config_dir", dest="grounded_dino_config_dir", type=str, default="GroundedSAM_demo/cfg/gdino",
                        help="Path of the dir containing the configuration files for 'GroundedDino'.")
    parser.add_argument("-u", "--grounded_dino_use_vitb", dest="grounded_dino_use_vitb", action="store_true", default=False,
                        help="Use the vit_b backbone for GroundingDino.")

    parser.add_argument("-m", "--box_threshold", dest="box_threshold", type=float, default=0.1,
                        help="The minimum confidence score of a bounding box to be used as prompt.")
    parser.add_argument("-n", "--text_threshold", dest="text_threshold", type=float, default=0.1,
                        help="The minimum confidence score that the bounding box class matches the phrase.")

    parser.add_argument("-y", "--use_yolo_sam", dest="use_yolo_sam", action="store_true", default=False,
                        help="Use Yolo implementation of SAM.")
    parser.add_argument("-s", "--sam_vit_model", dest="sam_vit_model", type=str, default="vit_b",
                        help="Which SAM model/backbone size to use.")
    parser.add_argument("-q", "--mask_threshold", dest="mask_threshold", type=float, default=0.01,
                        help="The minimum confidence score of a segmentation masks.")

    parser.add_argument("-p", "--prompt_text", dest="prompt_text", type=str, default="objects",
                        help="Prompt for the bounding box search.")

    parser.add_argument("-b", "--background", dest="background", choices=["rgb", "gray", "black", "white"],
                        default="gray",
                        help="Choose which background image for the renderings to use: 'rgb': scene image, 'gray': grayscale scene image, 'black'/'white': black/white background.")
    
    # F_Pose args
    code_dir = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')

    #TODO: add foundation pose args
    args = parser.parse_args()

    get_pose(device=args.device,
                              checkpoints_dir=args.checkpoints_dir,
                              grounded_dino_config_dir=args.grounded_dino_config_dir,
                              grounded_dino_use_vitb=args.grounded_dino_use_vitb,
                              box_threshold=args.box_threshold,
                              text_threshold=args.text_threshold,
                              use_yolo_sam=args.use_yolo_sam,
                              sam_vit_model=args.sam_vit_model,
                              mask_threshold=args.mask_threshold,
                              prompt_text=args.prompt_text,
                              background=args.background,
                              depth_path=args.depth_path,
                              mesh_path=args.mesh_path,
                              mesh_obj_id=args.mesh_obj_id,
                              intrinsics_path=args.intrinsics_path)
    


    # python3 -m taha_merge -i "GroundedSAM_demo/demo.mp4" -m 0.3 -n 0.3 -y -s "mobile_sam.pt" -q 0.01 -p "red cup" -b "gray"
    # python3 -m taha_merge -i "demo_data/YCB_Video/test/000048/rgb" --depth_path "demo_data/YCB_Video/test/000048/depth" -m 0.3 -n 0.3 -y -s "mobile_sam.pt" -q 0.01 -p "red cup" -b "gray" --mesh_path "demo_data/YCB_Video/models" --mesh_obj_id 14