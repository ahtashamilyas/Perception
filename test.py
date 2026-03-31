from estimater import *
from datareader import *
import argparse
from ultralytics import SAM
import torch
import os
import trimesh
from cv_bridge import CvBridge
import cv2
import numpy as np

parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--mesh_file', type=str, default='/home/student/Desktop/perception/FoundationPose/demo_data/cube/cube_mesh.obj')
parser.add_argument('--test_scene_dir', type=str, default='/home/student/Desktop/perception/FoundationPose/demo_data/cube_rgb.png')
parser.add_argument('--est_refine_iter', type=int, default=5)
parser.add_argument('--track_refine_iter', type=int, default=2)
parser.add_argument('--debug', type=int, default=1)
parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
args = parser.parse_args()
torch.cuda.set_device("cuda")

mesh = trimesh.load(args.mesh_file)
mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4))

scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()

bridge = CvBridge()
depth = None
color = None
cam_K = np.array([[1.066778e+03, 0.000000e+00, 3.129869e+02],
                  [0.000000e+00, 1.067487e+03, 2.413109e+02],
                  [0.000000e+00, 0.000000e+00, 1.000000e+00]])

seg_model = SAM("sam2.1_b.pt")

est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(), symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, refiner=refiner)


reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
H, W = reader.color_image.shape[:2]
color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)
depth = cv2.resize(depth, (W,H), interpolation=cv2.INTER_NEAREST)
depth[(depth < 0.1) | (depth >= np.inf)] = 0
mask = seg_model.predict(source=color, device="cuda") # TODO Mask Segmentation logic from color_based_mask.py
mask.save("masks.png") # TODO Mask Segmentation logic from color_based_mask.py
est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=symmetry_tfs, mesh=mesh)


# pose = est.track(K=cam_K, rgb=color, depth=depth, ob_mask=mask[0].masks.data.cpu().numpy()[0].astype(bool), init_pose=np.eye(4), iteration=args.track_refine_iter)