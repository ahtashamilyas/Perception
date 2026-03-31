# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
FoundationPose + GroundingDINO+SAM Multi-Object Detection for YCB Video

Simple integration that adds GroundingDINO+SAM detection to the original YCB video script.
Falls back to ground truth masks when GroundingSAM is not available.

Usage:
    # With ground truth masks (original behavior)
    python run_ycb_video_multi.py --detect_type mask
    
    # With GroundingDINO+SAM (requires installation)  
    python run_ycb_video_multi.py --detect_type grounding_sam
"""

from Utils import *
import json,uuid,joblib,os,sys,argparse
from datareader import *
from estimater import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')
import yaml
import time
import cv2
import trimesh
import numpy as np

from GroundedSAM_demo.grounded_sam import GroundedSAM


# # Try to import GroundingDINO and SAM
# try:
#     from groundingdino.util.inference import load_model, load_image, predict
#     from segment_anything import sam_model_registry, SamPredictor
#     GROUNDING_SAM_AVAILABLE = True
# except ImportError:
#     GROUNDING_SAM_AVAILABLE = False
#     print("GroundingDINO/SAM not available. Install with: pip install groundingdino segment_anything")

# class GroundingSAMDetector:
#     def __init__(self, device='cuda'):
#         self.device = device
#         self.grounding_model = None
#         self.sam_predictor = None
        
#         # YCB object names for detection prompts
#         self.ycb_names = {
#             1: "master chef can", 2: "cracker box", 3: "sugar box", 4: "tomato soup can",
#             5: "mustard bottle", 6: "tuna fish can", 7: "pudding box", 8: "gelatin box",
#             9: "potted meat can", 10: "banana", 11: "pitcher base", 12: "bleach cleanser",
#             13: "bowl", 14: "mug", 15: "power drill", 16: "wood block", 17: "scissors",
#             18: "large marker", 19: "large clamp", 20: "extra large clamp", 21: "foam brick"
#         }
        
#     def initialize(self):
#         if not GROUNDING_SAM_AVAILABLE:
#             return False
            
#         try:
#             # Load GroundingDINO
#             config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
#             checkpoint_path = "groundingdino_swint_ogc.pth"
#             self.grounding_model = load_model(config_path, checkpoint_path)
            
#             # Load SAM
#             sam_checkpoint = "sam_vit_h_4b8939.pth"
#             sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
#             sam.to(self.device)
#             self.sam_predictor = SamPredictor(sam)
            
#             print("GroundingSAM initialized successfully")
#             return True
#         except Exception as e:
#             print(f"Failed to initialize GroundingSAM: {e}")
#             return False
    
#     def detect_objects(self, image, object_ids):
#         """Detect YCB objects and return masks"""
#         if not self.grounding_model or not self.sam_predictor:
#             return {}
            
#         # Create text prompt from object names
#         object_names = [self.ycb_names.get(obj_id, f"object_{obj_id}") for obj_id in object_ids]
#         prompt = " . ".join(object_names)
        
#         try:
#             # GroundingDINO detection
#             image_pil, image_tensor = load_image(image)
#             boxes, confidences, labels = predict(
#                 model=self.grounding_model,
#                 image=image_tensor, 
#                 caption=prompt,
#                 box_threshold=0.35,
#                 text_threshold=0.25,
#                 device=self.device
#             )
            
#             if len(boxes) == 0:
#                 return {}
            
#             # SAM mask generation
#             self.sam_predictor.set_image(image)
#             masks = {}
            
#             for i, (box, label) in enumerate(zip(boxes, labels)):
#                 # Generate mask from bounding box
#                 mask, _, _ = self.sam_predictor.predict(box=box.reshape(1, -1), multimask_output=False)
                
#                 # Match label to object ID
#                 obj_id = self._match_label_to_id(label, object_ids)
#                 if obj_id and obj_id not in masks:  # Take first detection per object
#                     masks[obj_id] = mask[0]
                    
#             return masks
            
#         except Exception as e:
#             print(f"GroundingSAM detection failed: {e}")
#             return {}
    
#     def _match_label_to_id(self, label, target_ids):
#         """Simple matching of detected label to YCB object ID"""
#         label_lower = label.lower()
#         for obj_id in target_ids:
#             if obj_id in self.ycb_names:
#                 name_lower = self.ycb_names[obj_id].lower()
#                 if any(word in label_lower for word in name_lower.split()):
#                     return obj_id
#         return None

# # Global detector instance
# grounding_sam_detector = None

def get_mask(reader, i_frame, ob_id, detect_type):
    global grounding_sam_detector
    
    if detect_type == 'grounding_sam' and grounding_sam_detector:
        # Get GroundingSAM mask
        color = reader.get_color(i_frame)
        scene_objects = reader.get_instance_ids_in_image(i_frame)
        masks = grounding_sam_detector.detect_objects(color, scene_objects)
        
        if ob_id in masks:
            print(f"Using GroundingSAM mask for object {ob_id}")
            return masks[ob_id]
        else:
            print(f"GroundingSAM failed for object {ob_id}, using ground truth")
            # Fallback to ground truth
            mask = reader.get_mask(i_frame, ob_id, type='mask_visib')
            return mask > 0
    
    # Original mask methods
    elif detect_type=='box':
        mask = reader.get_mask(i_frame, ob_id)
        H,W = mask.shape[:2]
        vs,us = np.where(mask>0)
        if len(vs) == 0:
            return np.zeros((H,W), dtype=bool)
        umin = us.min()
        umax = us.max()
        vmin = vs.min()
        vmax = vs.max()
        valid = np.zeros((H,W), dtype=bool)
        valid[vmin:vmax,umin:umax] = 1
        return valid
    elif detect_type=='mask':
        mask = reader.get_mask(i_frame, ob_id, type='mask_visib')
        return mask>0
    elif detect_type=='cnos':
        mask = cv2.imread(reader.color_files[i_frame].replace('rgb','mask_cnos'), -1)
        return mask==ob_id
    else:
        raise RuntimeError(f"Unknown detect_type: {detect_type}")

def visualize_pose_result(color, pose, K, mesh, debug_dir=None, frame_id=None, show_window=True, mask=None, obj_id=None):
    """Visualize pose estimation result on the image"""
    try:
        vis = color.copy()
        
        # Draw mask overlay if provided
        if mask is not None:
            # Create colored overlay for mask
            mask_color = np.array([0, 255, 0])  # Green for single object
            if obj_id is not None:
                # Different colors for different objects
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                mask_color = colors[obj_id % len(colors)]
            
            overlay = np.zeros_like(vis)
            overlay[mask] = mask_color
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        # Get bounding box for visualization
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        
        center_pose = pose @ np.linalg.inv(to_origin)
        
        # Draw 3D bounding box and coordinate axes
        vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
        
        # Add pose and object information as text
        pos = pose[:3, 3]  # Extract translation
        if obj_id is not None:
            text = f"Object {obj_id}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
        else:
            text = f"Pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if frame_id is not None:
            cv2.putText(vis, f"Frame: {frame_id}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show in window if requested
        if show_window:
            cv2.imshow('FoundationPose Multi-Object', vis[...,::-1])  # Convert RGB to BGR for cv2
            cv2.waitKey(1)  # Non-blocking wait to update display
        
        # Save image if debug directory provided
        if debug_dir and frame_id:
            os.makedirs(f'{debug_dir}/pose_vis', exist_ok=True)
            cv2.imwrite(f'{debug_dir}/pose_vis/{frame_id}.png', vis[...,::-1])
        
        return vis
        
    except Exception as e:
        print(f"Visualization error: {e}")
        return color

def visualize_multi_object_results(color, poses_dict, K, meshes_dict, masks_dict=None, frame_id=None, debug_dir=None, show_window=True):
    """Visualize multiple objects in a single image"""
    try:
        vis = color.copy()
        
        # Draw all masks first
        if masks_dict:
            for i, (obj_id, mask) in enumerate(masks_dict.items()):
                if mask is not None and mask.any():
                    # Different colors for different objects
                    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), 
                             (255, 255, 100), (255, 100, 255), (100, 255, 255)]
                    mask_color = colors[i % len(colors)]
                    
                    overlay = np.zeros_like(vis)
                    overlay[mask] = mask_color
                    vis = cv2.addWeighted(vis, 0.8, overlay, 0.2, 0)
        
        # Draw 3D poses for each object
        text_y_offset = 30
        for i, (obj_id, pose) in enumerate(poses_dict.items()):
            if obj_id not in meshes_dict:
                continue
                
            mesh = meshes_dict[obj_id]
            
            # Get bounding box for 3D visualization
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
            center_pose = pose @ np.linalg.inv(to_origin)
            
            # Different colors for different objects
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                     (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            bbox_color = colors[i % len(colors)]
            
            # Draw 3D bounding box and coordinate axes
            vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.08, K=K, 
                               thickness=2, transparency=0, is_input_rgb=True)
            
            # Add object information
            pos = pose[:3, 3]
            text = f"Obj {obj_id}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            cv2.putText(vis, text, (10, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, bbox_color, 2)
            text_y_offset += 25
        
        # Add frame information
        if frame_id is not None:
            cv2.putText(vis, f"Frame: {frame_id} | Objects: {len(poses_dict)}", 
                       (10, vis.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Detection method info
        cv2.putText(vis, f"Detection: {opt.detect_type}", 
                   (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Show in window if requested
        if show_window:
            cv2.imshow('FoundationPose Multi-Object', vis[...,::-1])
            print(f"  Frame displayed. Press 'q'=quit, 'n'=next video, 's'=skip, or any key to continue...")
            key = cv2.waitKey(0) & 0xFF  # Wait for user input
            return key
        
        # Save combined visualization
        if debug_dir and frame_id:
            os.makedirs(f'{debug_dir}/multi_obj_vis', exist_ok=True)
            cv2.imwrite(f'{debug_dir}/multi_obj_vis/{frame_id}.png', vis[...,::-1])
        
        return vis
        
    except Exception as e:
        print(f"Multi-object visualization error: {e}")
        return color

def run_multi_object_pose_estimation():
    global grounding_sam_detector
    
    wp.force_load(device='cuda')
    video_dirs = sorted(glob.glob(f'{opt.ycbv_dir}/test/*'))[:opt.max_videos]
    res = NestDict()

    # Initialize GroundingSAM if needed
    if opt.detect_type == 'grounding_sam':
        # grounding_sam_detector = GroundingSAMDetector(device='cuda')
            # Initialize GroundedSAM
        grounded_sam = GroundedSAM.load_grounded_sam_model(
            checkpoints_dir="GroundedSAM_demo/checkpoints",
            grounded_dino_config_dir="GroundedSAM_demo/cfg/gdino",
            box_threshold=0.3,
            text_threshold=0.3,
            use_yolo_sam=False,
            sam_vit_model="mobile_sam.pt",
            mask_threshold=0.01,
            prompt_text="red mug",
            segmentor_width_size=None,
            device=None
        )
        if not grounded_sam is None:
            print("Falling back to ground truth masks")
            opt.detect_type = 'mask'
            grounding_sam_detector = None

    reader_tmp = YcbVideoReader(video_dirs[0])
    glctx = dr.RasterizeCudaContext()
    mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4))
    est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(), 
                         symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, 
                         debug_dir=opt.debug_dir, debug=opt.debug)

    # Get target objects
    if opt.target_objects:
        target_ob_ids = [int(x) for x in opt.target_objects.split(',')]
    else:
        target_ob_ids = reader_tmp.ob_ids

    print(f"Target objects: {target_ob_ids}")
    
    if opt.show_visualization:
        cv2.namedWindow('FoundationPose Multi-Object', cv2.WINDOW_AUTOSIZE)
        print("\n" + "="*60)
        print("VISUALIZATION CONTROLS:")
        print("  'q' - Quit the program")
        print("  'n' - Skip to next video")
        print("  's' - Skip current frame")
        print("  Any other key - Continue to next frame")
        print("="*60 + "\n")

    # Timing accumulators
    frame_object_loop_times = []  # time spent only in per-object loop
    frame_total_times = []        # total frame processing time (excluding visualization wait for user key)

    # Process each video
    for video_dir in video_dirs:
        print(f"\nProcessing video: {os.path.basename(video_dir)}")
        reader = YcbVideoReader(video_dir, zfar=1.5)
        video_id = reader.get_video_id()
        
        # Find keyframes
        keyframes = [i for i in range(len(reader.color_files)) if reader.is_keyframe(i)][:opt.max_keyframes]
        print(f"Processing {len(keyframes)} keyframes")
        
    for i_frame in keyframes:
            id_str = reader.id_strs[i_frame]
            print(f"\n--- Processing Frame {i_frame+1}/{len(keyframes)}: {id_str} ---")
            
            color = reader.get_color(i_frame)
            depth = reader.get_depth(i_frame) 
            scene_ob_ids = reader.get_instance_ids_in_image(i_frame)
            
            # Process each target object in the scene
            present_objects = [obj_id for obj_id in target_ob_ids if obj_id in scene_ob_ids]
            if not present_objects:
                print(f"No target objects found in frame {id_str}")
                if opt.show_visualization:
                    cv2.imshow('FoundationPose Multi-Object', color[...,::-1])
                    key = cv2.waitKey(500) & 0xFF  # Show for 0.5 seconds
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        return
                continue
                
            print(f"Found {len(present_objects)} target objects: {present_objects}")
            
            # Collect results for all objects in this frame
            frame_poses = {}
            frame_meshes = {}
            frame_masks = {}
            t_frame_start = time.time()
            t_obj_loop_start = time.time()
            
            for ob_id in present_objects:
                # Get mesh and setup estimator
                try:
                    if opt.use_reconstructed_mesh:
                        mesh = reader.get_reconstructed_mesh(ob_id, ref_view_dir=opt.ref_view_dir)
                    else:
                        mesh = reader.get_gt_mesh(ob_id)
                    symmetry_tfs = reader.symmetry_tfs[ob_id]
                    
                    est.reset_object(model_pts=mesh.vertices.copy(), 
                                   model_normals=mesh.vertex_normals.copy(), 
                                   symmetry_tfs=symmetry_tfs, mesh=mesh)
                    
                    # Get mask
                    ob_mask = get_mask(reader, i_frame, ob_id, opt.detect_type)
                    
                    if not ob_mask.any():
                        print(f"No valid mask for object {ob_id}")
                        continue
                    
                    # Run pose estimation
                    est.gt_pose = reader.get_gt_pose(i_frame, ob_id)
                    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=ob_id, iteration=5)
                    
                    # Store results
                    res[video_id][id_str][ob_id] = pose
                    frame_poses[ob_id] = pose
                    frame_meshes[ob_id] = mesh
                    frame_masks[ob_id] = ob_mask
                    
                    print(f"Estimated pose for object {ob_id}")
                    
                except Exception as e:
                    print(f"Failed processing object {ob_id}: {e}")
                    continue

            # Record timing for object loop
            t_obj_loop_end = time.time()
            obj_loop_dt = t_obj_loop_end - t_obj_loop_start
            frame_object_loop_times.append(obj_loop_dt)
            # Note: exclude blocking visualization wait ('cv2.waitKey(0)') from total time to keep numbers comparable.
            t_after_processing = t_obj_loop_end
            
            # Visualize all objects together (show immediately after processing)
            if opt.show_visualization and frame_poses:
                print(f"Showing visualization for frame {id_str} with {len(frame_poses)} objects...")
                key = visualize_multi_object_results(
                    color=color,
                    poses_dict=frame_poses,
                    K=reader.K,
                    meshes_dict=frame_meshes,
                    masks_dict=frame_masks,
                    frame_id=f"video_{video_id}_frame_{id_str}",
                    debug_dir=opt.debug_dir if opt.debug >= 1 else None,
                    show_window=True
                )
                
                # Check for quit or next commands
                if key == ord('q'):
                    print("Quit requested by user")
                    cv2.destroyAllWindows()
                    return
                elif key == ord('n'):
                    print("Next video requested")
                    break
                elif key == ord('s'):
                    print("Skipping current frame")
                    # Record frame total up to processing end (excluding visualization wait since we already blocked)
                    frame_total_times.append(t_after_processing - t_frame_start)
                    continue
            elif opt.show_visualization and not frame_poses:
                print(f"No objects detected in frame {id_str}, skipping visualization...")
                # Show original image for 1 second
                cv2.imshow('FoundationPose Multi-Object', color[...,::-1])
                key = cv2.waitKey(1000) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
            # Record frame total time (exclude user visualization wait; if a visualization occurred we use processing end timestamp)
            frame_total_times.append(t_after_processing - t_frame_start)
            print(f"Timing: object_loop={obj_loop_dt:.3f}s, frame_total(no-vis-wait)={frame_total_times[-1]:.3f}s")
    
    # Clean up and save
    if opt.show_visualization:
        cv2.destroyAllWindows()
    
    output_file = f'{opt.debug_dir}/ycbv_multi_object_res.yml'
    print(f"\nSaving results to: {output_file}")
    os.makedirs(opt.debug_dir, exist_ok=True)
    with open(output_file, 'w') as ff:
        yaml.safe_dump(make_yaml_dumpable(res), ff)
    
    # Timing summary
    if frame_object_loop_times:
        avg_obj_loop = sum(frame_object_loop_times)/len(frame_object_loop_times)
        avg_frame_total = sum(frame_total_times)/len(frame_total_times)
        print("\nTiming Summary (sequential multi.py):")
        print(f"  Frames processed: {len(frame_object_loop_times)}")
        print(f"  Avg per-frame object loop time: {avg_obj_loop:.3f}s")
        print(f"  Avg per-frame total (no visualization wait): {avg_frame_total:.3f}s")
        print(f"  Min/Max object loop: {min(frame_object_loop_times):.3f}s / {max(frame_object_loop_times):.3f}s")
    print("Multi-object pose estimation completed!")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    
    parser.add_argument('--ycbv_dir', type=str, default="/home/student/Desktop/perception/FoundationPose/demo_data/YCB_Video")
    parser.add_argument('--detect_type', type=str, default='grounding_sam', choices=['mask', 'box', 'cnos', 'grounding_sam'])
    parser.add_argument('--target_objects', type=str, default='', help="Comma-separated object IDs (e.g., '2,5,6')")
    parser.add_argument('--max_videos', type=int, default=2)
    parser.add_argument('--max_keyframes', type=int, default=5) 
    parser.add_argument('--use_reconstructed_mesh', type=int, default=0)
    parser.add_argument('--ref_view_dir', type=str, default="")
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    parser.add_argument('--show_visualization', type=int, default=1)
    
    opt = parser.parse_args()
    os.environ["YCB_VIDEO_DIR"] = opt.ycbv_dir
    set_seed(0)
    
    run_multi_object_pose_estimation()
