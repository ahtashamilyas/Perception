"""
Threaded multi-object pose estimation based on multi.py.
Per-object processing loop is parallelized with ThreadPoolExecutor.
Avoids sharing a FoundationPose estimator or CUDA raster context across threads.
Supports multiple objects in YCB-Video using existing mask/box/cnos detectors.
"""

from Utils import *  # noqa
import json,uuid,joblib,os,sys,argparse  # noqa
from datareader import *  # noqa
from estimater import *  # noqa
import yaml
import cv2
import trimesh
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')

 

def get_mask(reader, i_frame, ob_id, detect_type):
    if detect_type=='box':
        mask = reader.get_mask(i_frame, ob_id)
        H,W = mask.shape[:2]
        vs,us = np.where(mask>0)
        if len(vs) == 0:
            return np.zeros((H,W), dtype=bool)
        umin, umax = us.min(), us.max()
        vmin, vmax = vs.min(), vs.max()
        valid = np.zeros((H,W), dtype=bool)
        valid[vmin:vmax, umin:umax] = 1
        return valid
    elif detect_type=='mask':
        mask = reader.get_mask(i_frame, ob_id, type='mask_visib')
        return mask>0
    elif detect_type=='cnos':
        mask = cv2.imread(reader.color_files[i_frame].replace('rgb','mask_cnos'), -1)
        return mask==ob_id
    else:
        raise RuntimeError(f"Unknown detect_type: {detect_type}")


def visualize_multi_object_results(color, poses_dict, K, meshes_dict, masks_dict=None, frame_id=None, debug_dir=None, show_window=True):
    # (Simplified copy from original)
    try:
        vis = color.copy()
        if masks_dict:
            for i,(obj_id, mask) in enumerate(masks_dict.items()):
                if mask is not None and mask.any():
                    colors = [(255,100,100),(100,255,100),(100,100,255),(255,255,100),(255,100,255),(100,255,255)]
                    mask_color = colors[i % len(colors)]
                    overlay = np.zeros_like(vis)
                    overlay[mask] = mask_color
                    vis = cv2.addWeighted(vis,0.8,overlay,0.2,0)
        text_y_offset = 30
        for i,(obj_id, pose) in enumerate(poses_dict.items()):
            mesh = meshes_dict.get(obj_id)
            if mesh is None: continue
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.08, K=K, thickness=2, transparency=0, is_input_rgb=True)
            pos = pose[:3,3]
            txt = f"Obj {obj_id}: ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})"
            cv2.putText(vis, txt, (10, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            text_y_offset += 22
        if frame_id is not None:
            cv2.putText(vis, f"Frame: {frame_id} | Objects: {len(poses_dict)}", (10, vis.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
        cv2.putText(vis, f"Detection: {opt.detect_type}", (10, vis.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
        if show_window:
            cv2.imshow('FoundationPose Multi-Object (threaded)', vis[...,::-1])
            key = cv2.waitKey(0) & 0xFF
            return key
        return vis
    except Exception as e:
        print(f"Visualization error: {e}")
        return color


def run_multi_object_pose_estimation_threaded():
    wp.force_load(device='cuda')
    video_dirs = sorted(glob.glob(f'{opt.ycbv_dir}/test/*'))[:opt.max_videos]
    res = NestDict()

    reader_tmp = YcbVideoReader(video_dirs[0])
    glctx = dr.RasterizeCudaContext()
    # Use a mutable Trimesh (not a primitive) to avoid immutable vertices error
    mesh_tmp = trimesh.creation.box(extents=np.ones((3,)))
    est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(),
                         symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx,
                         debug_dir=opt.debug_dir, debug=opt.debug)

    if opt.target_objects:
        target_ob_ids = [int(x) for x in opt.target_objects.split(',')]
    else:
        target_ob_ids = reader_tmp.ob_ids
    print(f"Target objects: {target_ob_ids}")

    if opt.show_visualization:
        cv2.namedWindow('FoundationPose Multi-Object (threaded)', cv2.WINDOW_AUTOSIZE)

    # Timing accumulators
    frame_object_loop_times = []  # time spent in parallel object loop only
    frame_total_times = []        # total frame processing time excluding visualization blocking wait

    for video_dir in video_dirs:
        print(f"\nProcessing video: {os.path.basename(video_dir)}")
        reader = YcbVideoReader(video_dir, zfar=1.5)
        video_id = reader.get_video_id()
        keyframes = [i for i in range(len(reader.color_files)) if reader.is_keyframe(i)][:opt.max_keyframes]
        print(f"Processing {len(keyframes)} keyframes")
        for i_frame in keyframes:
            id_str = reader.id_strs[i_frame]
            print(f"\n--- Frame {i_frame+1}/{len(keyframes)}: {id_str} ---")
            color = reader.get_color(i_frame)
            depth = reader.get_depth(i_frame)
            scene_ob_ids = reader.get_instance_ids_in_image(i_frame)
            present_objects = [obj_id for obj_id in target_ob_ids if obj_id in scene_ob_ids]
            if not present_objects:
                print('No target objects in frame.')
                continue
            print(f"Found {len(present_objects)} target objects: {present_objects}")

            t_frame_start = time.time()

            frame_poses, frame_meshes, frame_masks = {}, {}, {}

            def process_object(ob_id):
                try:
                    if opt.use_reconstructed_mesh:
                        mesh = reader.get_reconstructed_mesh(ob_id, ref_view_dir=opt.ref_view_dir)
                    else:
                        mesh = reader.get_gt_mesh(ob_id)
                    symmetry_tfs = reader.symmetry_tfs[ob_id]
                    # Create per-thread context & estimator
                    glctx_local = dr.RasterizeCudaContext()
                    est_local = FoundationPose(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(),
                                               symmetry_tfs=symmetry_tfs, mesh=mesh, scorer=None, refiner=None,
                                               glctx=glctx_local, debug_dir=opt.debug_dir, debug=opt.debug)
                    # Mask
                    ob_mask = get_mask(reader, i_frame, ob_id, opt.detect_type)
                    if ob_mask is None or not ob_mask.any():
                        return (ob_id, None, None, None, 'empty mask')
                    # Pose estimation
                    pose = est_local.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=ob_id, iteration=5)
                    print(f"Object {ob_id} pose:\n{pose}")
                    return (ob_id, pose, mesh, ob_mask, None)
                except Exception as e:
                    return (ob_id, None, None, None, str(e))

            t0 = time.time()
            max_workers = min(len(present_objects), opt.num_workers)
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                fut_map = {ex.submit(process_object, oid): oid for oid in present_objects}
                for fut in as_completed(fut_map):
                    results.append(fut.result())
            dt = time.time() - t0
            print(f"Parallel processing took {dt:.3f}s with {max_workers} workers")
            frame_object_loop_times.append(dt)

            # Integrate results on main thread
            for ob_id, pose, mesh, ob_mask, err in results:
                if err or pose is None:
                    print(f"Failed processing object {ob_id}: {err}")
                    continue
                res[video_id][id_str][ob_id] = pose
                frame_poses[ob_id] = pose
                frame_meshes[ob_id] = mesh
                frame_masks[ob_id] = ob_mask
                print(f"Estimated pose for object {ob_id}")

            t_after_processing = time.time()
            if opt.show_visualization and frame_poses:
                key = visualize_multi_object_results(color, frame_poses, reader.K, frame_meshes, frame_masks,
                                                      frame_id=f"video_{video_id}_frame_{id_str}",
                                                      debug_dir=opt.debug_dir if opt.debug>=1 else None,
                                                      show_window=True)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                elif key == ord('n'):
                    break
                elif key == ord('s'):
                    continue
            # Record timing for frame (excluding visualization blocking time)
            frame_total_times.append(t_after_processing - t_frame_start)
            print(f"Timing: object_loop_parallel={dt:.3f}s, frame_total(no-vis-wait)={frame_total_times[-1]:.3f}s")

    if opt.show_visualization:
        cv2.destroyAllWindows()
    output_file = f'{opt.debug_dir}/ycbv_multi_object_res_threaded.yml'
    print(f"\nSaving results to: {output_file}")
    os.makedirs(opt.debug_dir, exist_ok=True)
    with open(output_file, 'w') as ff:
        yaml.safe_dump(make_yaml_dumpable(res), ff)
    if frame_object_loop_times:
        avg_obj_loop = sum(frame_object_loop_times)/len(frame_object_loop_times)
        avg_frame_total = sum(frame_total_times)/len(frame_total_times)
        print("\nTiming Summary (threaded multi_thread.py):")
        print(f"  Frames processed: {len(frame_object_loop_times)}")
        print(f"  Avg per-frame object loop time: {avg_obj_loop:.3f}s")
        print(f"  Avg per-frame total (no visualization wait): {avg_frame_total:.3f}s")
        print(f"  Min/Max object loop: {min(frame_object_loop_times):.3f}s / {max(frame_object_loop_times):.3f}s")
    print('Threaded multi-object pose estimation completed!')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ycbv_dir', type=str, default="/home/student/Desktop/perception/FoundationPose/demo_data/YCB_Video")
    parser.add_argument('--detect_type', type=str, default='mask', choices=['mask','box','cnos'])
    parser.add_argument('--target_objects', type=str, default='')
    parser.add_argument('--max_videos', type=int, default=2)
    parser.add_argument('--max_keyframes', type=int, default=5)
    parser.add_argument('--use_reconstructed_mesh', type=int, default=0)
    parser.add_argument('--ref_view_dir', type=str, default="")
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    parser.add_argument('--show_visualization', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4, help='Max threads for per-frame object processing')
    opt = parser.parse_args()
    os.environ['YCB_VIDEO_DIR'] = opt.ycbv_dir
    set_seed(0)
    run_multi_object_pose_estimation_threaded()
