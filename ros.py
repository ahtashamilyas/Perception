# Optional ROS2 (Jazzy) imports; loaded only when --use_ros2 is enabled
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
except Exception:
    rclpy = None
    Node = object
    Image = None
    CameraInfo = None
    CvBridge = None





# -------------------- ROS2 (Jazzy) integration --------------------
class Ros2PoseNode(Node):
    """ROS2 Jazzy node to read RGB, depth, and camera intrinsics from RealSense topics
    and run GroundedSAM + FoundationPose pose estimation. No ROS outputs for now.
    """
    def __init__(self, grounded_sam, est, loaded_mesh, prompt_text,
                 rgb_topic: str, depth_topic: str, camera_info_topic: str,
                 background: str = "gray"):
        super().__init__('foundationpose_ros2_node')

        if CvBridge is None:
            raise RuntimeError("cv_bridge is not available. Please install ros-$ROS_DISTRO-cv-bridge")

        self.bridge = CvBridge()
        self.grounded_sam = grounded_sam
        self.est = est
        self.mesh = loaded_mesh
        self.prompt_text = prompt_text
        self.background = background

        self.cam_K = None
        self.color_image = None  # RGB
        self.depth_image = None  # float32 in meters

        # Colors for mask overlays
        self.colors = distinctipy.get_colors(50)

        # Subscriptions
        self.create_subscription(Image, rgb_topic, self._rgb_cb, 10)
        self.create_subscription(Image, depth_topic, self._depth_cb, 10)
        self.create_subscription(CameraInfo, camera_info_topic, self._camera_info_cb, 10)

        # UI
        cv2.namedWindow("FoundationPose ROS2", cv2.WINDOW_NORMAL)

    def _camera_info_cb(self, msg: CameraInfo):
        # CameraInfo.k is a 9-element row-major intrinsic matrix
        try:
            K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
            self.cam_K = K
        except Exception as e:
            self.get_logger().warn(f"Failed to parse CameraInfo: {e}")

    def _rgb_cb(self, msg: Image):
        # Convert to RGB8 numpy
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            # Ensure contiguous uint8 RGB
            img = np.asarray(img, dtype=np.uint8)
            img = np.ascontiguousarray(img)
            self.color_image = img
        except Exception as e:
            self.get_logger().warn(f"RGB conversion failed: {e}")

    def _depth_cb(self, msg: Image):
        # Convert depth to float32 meters; honor encoding
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth.dtype == np.uint16:
                depth_m = depth.astype(np.float32) * 1e-3  # 16UC1 in mm
            else:
                # Assume meters already (e.g., 32FC1)
                depth_m = depth.astype(np.float32)
            # Sanitize
            depth_m[depth_m < 0.001] = 0
            depth_m[depth_m > ZFAR] = 0
            self.depth_image = depth_m
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")
            self.depth_image = None

        # Process when we have synchronized data
        self._process_if_ready()

    def _process_if_ready(self):
        if self.color_image is None or self.depth_image is None or self.cam_K is None:
            return

        rgb_frame = self.color_image
        depth = self.depth_image

        # Detection via GroundedSAM
        detections = self.grounded_sam.generate_masks(rgb_frame)

        masks = None
        frame = rgb_frame
        if detections:
            masks_arr = detections["masks"].squeeze(1).cpu().numpy()
            masks_scores = detections["masks_scores"].cpu().numpy()
            sort_indicis = np.argsort(-masks_scores)
            masks_arr = masks_arr[sort_indicis]
            masks_scores = masks_scores[sort_indicis]

            # Light morphology cleanup
            kernel = np.ones((3, 3), np.uint8)
            cleaned_masks = []
            for m in masks_arr:
                m_u8 = (m.astype(np.uint8) * 255)
                m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN, kernel)
                m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, kernel)
                cleaned_masks.append(m_u8.astype(bool))
            masks = np.array(cleaned_masks)

            # Compose background
            if self.background == "rgb":
                frame = rgb_frame.copy()
            elif self.background == "gray":
                frame = cv2.cvtColor(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            elif self.background == "black":
                frame = np.zeros_like(rgb_frame, dtype=np.uint8)
            elif self.background == "white":
                frame = np.ones_like(rgb_frame, dtype=np.uint8)

            alpha = 0.3
            for mask_idx, mask in enumerate(masks):
                r = int(255 * self.colors[mask_idx][0])
                g = int(255 * self.colors[mask_idx][1])
                b = int(255 * self.colors[mask_idx][2])
                frame[mask, 0] = alpha * r + (1 - alpha) * frame[mask, 0]
                frame[mask, 1] = alpha * g + (1 - alpha) * frame[mask, 1]
                frame[mask, 2] = alpha * b + (1 - alpha) * frame[mask, 2]

        # Pose estimation (single most confident mask)
        out = None
        if masks is not None and masks.shape[0] > 0 and self.mesh is not None:
            most_confident_mask = masks[0]
            try:
                out = run_pose_estimation_worker(
                    most_confident_mask, self.est, self.cam_K,
                    ob_id=None, device=0, depth_image=depth, rgb_image=rgb_frame
                )
            except Exception as e:
                self.get_logger().warn(f"Pose estimation error: {e}")
                out = None

        # Visualization overlays
        disp_frame = frame if masks is not None else rgb_frame
        if out is not None and self.mesh is not None:
            try:
                pose_np = out.detach().cpu().numpy() if hasattr(out, 'detach') else (
                    out.cpu().numpy() if hasattr(out, 'cpu') else np.asarray(out))
            except Exception:
                pose_np = np.asarray(out)
            try:
                vis_img = visualize_pose_result(
                    color=disp_frame,
                    pose=pose_np,
                    K=self.cam_K,
                    mesh=self.mesh,
                    debug_dir=None,
                    frame_id=None,
                    show_window=False,
                )
                disp_frame = vis_img
            except Exception as e:
                self.get_logger().warn(f"Visualization error: {e}")

        cv2.imshow("FoundationPose ROS2", cv2.cvtColor(disp_frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)


def run_ros2_pose_estimation(
    rgb_topic: str,
    depth_topic: str,
    camera_info_topic: str,
    checkpoints_dir: str,
    grounded_dino_config_dir: str,
    grounded_dino_use_vitb: bool,
    box_threshold: float,
    text_threshold: float,
    use_yolo_sam: bool,
    sam_vit_model: str,
    mask_threshold: float,
    prompt_text: str,
    background: str,
    mesh_path: str,
    mesh_obj_id: int,
    debug_dir: str,
    debug: int,
):
    if rclpy is None:
        raise RuntimeError("rclpy is not available. Source /opt/ros/jazzy/setup.bash in this shell.")

    grounded_sam, est = load_models(
        checkpoints_dir=checkpoints_dir,
        grounded_dino_config_dir=grounded_dino_config_dir,
        grounded_dino_use_vitb=grounded_dino_use_vitb,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        use_yolo_sam=use_yolo_sam,
        sam_vit_model=sam_vit_model,
        mask_threshold=mask_threshold,
        prompt_text=prompt_text,
        debug_dir=debug_dir,
        debug=debug,
    )

    loaded_mesh = load_mesh(mesh_path=mesh_path, mesh_obj_id=mesh_obj_id, est=est)

    rclpy.init()
    node = Ros2PoseNode(
        grounded_sam=grounded_sam,
        est=est,
        loaded_mesh=loaded_mesh,
        prompt_text=prompt_text,
        rgb_topic=rgb_topic,
        depth_topic=depth_topic,
        camera_info_topic=camera_info_topic,
        background=background,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()











 # ROS2 (Jazzy) args
    parser.add_argument('--use_ros2', action='store_true', default=False,
                        help='Enable ROS2 Jazzy mode to read RealSense topics instead of files.')
    parser.add_argument('--ros_rgb_topic', type=str,
                        default='/tracy_camera/camera/camera/color/image_raw',
                        help='RGB image topic name.')
    parser.add_argument('--ros_depth_topic', type=str,
                        default='/tracy_camera/camera/camera/depth/image_rect_raw',
                        help='Depth image topic name (aligned to color).')
    parser.add_argument('--ros_camera_info_topic', type=str,
                        default='/tracy_camera/camera/camera/color/camera_info',
                        help='CameraInfo topic for intrinsics.')

    # Parse CLI args (no hard-coded test list)
    args = parser.parse_args()
    
    if args.use_ros2:
        # Run ROS2 Jazzy subscriber-based estimation from RealSense topics
        run_ros2_pose_estimation(
            rgb_topic=args.ros_rgb_topic,
            depth_topic=args.ros_depth_topic,
            camera_info_topic=args.ros_camera_info_topic,
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
            mesh_path=args.mesh_path,
            mesh_obj_id=args.mesh_obj_id,
            debug_dir=args.debug_dir,
            debug=args.debug,
        )
    else:
        # Default file/dataset mode
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
                 intrinsics_path=args.intrinsics_path,
                 scene_gt_path=args.scene_gt_path)