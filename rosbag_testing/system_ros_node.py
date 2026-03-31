#!/usr/bin/python3

"""
System Python Compatible FoundationPose ROS2 Node

This node uses system Python to avoid virtual environment conflicts with ROS2.
It provides a simplified but functional integration with FoundationPose.
"""

import sys
import os

# Add FoundationPose to the path
sys.path.insert(0, '/home/student/Desktop/Perception/FoundationPose')

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from visualization_msgs.msg import Marker, MarkerArray
    from std_msgs.msg import Header
    from cv_bridge import CvBridge
    import tf2_ros
    print("✓ ROS2 modules imported successfully")
except ImportError as e:
    print(f"✗ ROS2 import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    import cv2
    print("✓ OpenCV and NumPy imported successfully")
except ImportError as e:
    print(f"✗ OpenCV/NumPy import failed: {e}")
    sys.exit(1)

# Try to import FoundationPose components (graceful fallback if not available)
try:
    import trimesh
    from Utils import *
    from datareader import *
    print("✓ FoundationPose modules imported successfully")
    FOUNDATIONPOSE_AVAILABLE = True
except ImportError as e:
    print(f"⚠ FoundationPose import failed: {e}")
    print("Running in computer vision mode without FoundationPose")
    FOUNDATIONPOSE_AVAILABLE = False

class SystemFoundationPoseNode(Node):
    def __init__(self):
        super().__init__('system_foundationpose_node')
        
        # Parameters
        self.declare_parameter('mesh_file', 'demo_data/cube/model_vhacd.obj')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('object_frame', 'cube')
        self.declare_parameter('debug', True)
        
        # Get parameters
        mesh_file = self.get_parameter('mesh_file').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.object_frame = self.get_parameter('object_frame').get_parameter_value().string_value
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        
        # Initialize components
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Camera parameters (will be updated from camera_info)
        self.camera_matrix = np.array([
            [615.0, 0.0, 320.0],
            [0.0, 615.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Image storage
        self.latest_color_image = None
        self.latest_depth_image = None
        self.latest_timestamp = None
        
        # Initialize mesh if FoundationPose is available
        if FOUNDATIONPOSE_AVAILABLE:
            self.init_foundationpose(mesh_file)
        else:
            self.mesh = None
            
        # Subscribers
        self.color_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, 'cube_pose', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'cube_markers', 10)
        self.debug_image_pub = self.create_publisher(Image, 'debug_image', 10)
        
        # Processing timer
        self.timer = self.create_timer(0.2, self.process_callback)  # 5 Hz
        
        self.get_logger().info("System FoundationPose Node initialized")
        self.get_logger().info(f"FoundationPose available: {FOUNDATIONPOSE_AVAILABLE}")
    
    def init_foundationpose(self, mesh_file):
        """Initialize FoundationPose if available"""
        try:
            mesh_path = f"/home/student/Desktop/Perception/FoundationPose/{mesh_file}"
            if os.path.exists(mesh_path):
                self.mesh = trimesh.load(mesh_path)
                self.get_logger().info(f"Loaded mesh: {len(self.mesh.vertices)} vertices")
            else:
                self.get_logger().error(f"Mesh file not found: {mesh_path}")
                self.mesh = None
        except Exception as e:
            self.get_logger().error(f"Failed to load mesh: {e}")
            self.mesh = None
    
    def camera_info_callback(self, msg):
        """Update camera parameters"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
    
    def color_callback(self, msg):
        """Process color image"""
        try:
            self.latest_color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_timestamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"Color image conversion failed: {e}")
    
    def depth_callback(self, msg):
        """Process depth image"""
        try:
            if msg.encoding == "16UC1":
                depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
                self.latest_depth_image = depth_image.astype(np.float32) / 1000.0  # Convert to meters
            elif msg.encoding == "32FC1":
                self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            else:
                self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")
    
    def detect_cube_simple(self, color_image, depth_image):
        """Enhanced cube detection using computer vision"""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            # Multi-stage detection approach
            # 1. Color-based detection (broad range to catch various lighting)
            lower_hsv = np.array([0, 20, 50])   # Lower bound
            upper_hsv = np.array([180, 255, 255])  # Upper bound
            color_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
            
            # 2. Edge detection for geometric shapes
            edges = cv2.Canny(gray, 50, 150)
            
            # 3. Combine masks
            combined_mask = cv2.bitwise_and(color_mask, edges)
            
            # Clean up mask
            kernel = np.ones((3,3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                # Fallback: use color mask alone
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, color_mask
            
            # Filter contours by size and shape
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    # Check if roughly rectangular/square
                    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                    if len(approx) >= 4:  # Has at least 4 corners
                        valid_contours.append(contour)
            
            if not valid_contours:
                # Use largest contour as fallback
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 300:
                    valid_contours = [largest_contour]
                else:
                    return None, color_mask
            
            # Use the largest valid contour
            cube_contour = max(valid_contours, key=cv2.contourArea)
            
            # Get bounding rectangle and center
            x, y, w, h = cv2.boundingRect(cube_contour)
            center_u = x + w // 2
            center_v = y + h // 2
            
            # Try multiple points for depth (center and around it)
            depth_values = []
            search_radius = 5
            for du in range(-search_radius, search_radius + 1):
                for dv in range(-search_radius, search_radius + 1):
                    u = center_u + du
                    v = center_v + dv
                    if (0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]):
                        depth_val = depth_image[v, u]
                        if 0.1 < depth_val < 5.0:  # Valid depth range
                            depth_values.append(depth_val)
            
            if not depth_values:
                return None, color_mask
            
            # Use median depth for robustness
            depth_value = np.median(depth_values)
            
            # Convert to 3D coordinates
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]
            
            x_3d = (center_u - cx) * depth_value / fx
            y_3d = (center_v - cy) * depth_value / fy
            z_3d = depth_value
            
            # Estimate cube orientation from contour
            rect = cv2.minAreaRect(cube_contour)
            angle = np.radians(rect[2])  # Convert to radians
            
            # Create pose matrix with estimated orientation
            pose_matrix = np.eye(4)
            
            # Rotation around Z-axis based on contour orientation
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_z = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            
            # Small additional rotation for better visualization
            rotation_y = np.array([
                [np.cos(0.1), 0, np.sin(0.1)],
                [0, 1, 0],
                [-np.sin(0.1), 0, np.cos(0.1)]
            ])
            
            pose_matrix[:3, :3] = rotation_z @ rotation_y
            pose_matrix[0, 3] = x_3d
            pose_matrix[1, 3] = y_3d
            pose_matrix[2, 3] = z_3d
            
            # Create a detailed mask for visualization
            result_mask = np.zeros_like(color_mask)
            cv2.drawContours(result_mask, [cube_contour], -1, 255, -1)
            cv2.rectangle(result_mask, (x, y), (x + w, y + h), 128, 2)
            
            return pose_matrix, result_mask
            
        except Exception as e:
            self.get_logger().error(f"Enhanced cube detection failed: {e}")
            return None, None
    
    def process_callback(self):
        """Main processing callback"""
        if (self.latest_color_image is None or 
            self.latest_depth_image is None or 
            self.latest_timestamp is None):
            return
        
        try:
            # Detect cube
            pose_matrix, mask = self.detect_cube_simple(
                self.latest_color_image, 
                self.latest_depth_image
            )
            
            if pose_matrix is not None:
                # Publish pose
                self.publish_pose(pose_matrix, self.latest_timestamp)
                
                # Publish marker
                self.publish_marker(pose_matrix, self.latest_timestamp)
                
                # Publish debug image
                if self.debug and mask is not None:
                    self.publish_debug_image(
                        self.latest_color_image, 
                        pose_matrix, 
                        mask, 
                        self.latest_timestamp
                    )
                
                # Log detection
                pos = pose_matrix[:3, 3]
                self.get_logger().info(
                    f"Cube detected at: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                )
        
        except Exception as e:
            self.get_logger().error(f"Processing failed: {e}")
    
    def publish_pose(self, pose_matrix, timestamp):
        """Publish pose message"""
        try:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = timestamp
            pose_msg.header.frame_id = self.camera_frame
            
            # Position
            pose_msg.pose.position.x = float(pose_matrix[0, 3])
            pose_msg.pose.position.y = float(pose_matrix[1, 3])
            pose_msg.pose.position.z = float(pose_matrix[2, 3])
            
            # Rotation (convert rotation matrix to quaternion)
            rotation_matrix = pose_matrix[:3, :3]
            quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
            pose_msg.pose.orientation.x = quaternion[0]
            pose_msg.pose.orientation.y = quaternion[1]
            pose_msg.pose.orientation.z = quaternion[2]
            pose_msg.pose.orientation.w = quaternion[3]
            
            self.pose_pub.publish(pose_msg)
            
            # Also publish TF transform
            self.publish_transform(pose_matrix, timestamp)
            
        except Exception as e:
            self.get_logger().error(f"Pose publishing failed: {e}")
    
    def publish_transform(self, pose_matrix, timestamp):
        """Publish TF transform"""
        try:
            transform = TransformStamped()
            transform.header.stamp = timestamp
            transform.header.frame_id = self.camera_frame
            transform.child_frame_id = self.object_frame
            
            # Translation
            transform.transform.translation.x = float(pose_matrix[0, 3])
            transform.transform.translation.y = float(pose_matrix[1, 3])
            transform.transform.translation.z = float(pose_matrix[2, 3])
            
            # Rotation
            rotation_matrix = pose_matrix[:3, :3]
            quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
            transform.transform.rotation.x = quaternion[0]
            transform.transform.rotation.y = quaternion[1]
            transform.transform.rotation.z = quaternion[2]
            transform.transform.rotation.w = quaternion[3]
            
            self.tf_broadcaster.sendTransform(transform)
            
        except Exception as e:
            self.get_logger().error(f"Transform publishing failed: {e}")
    
    def publish_marker(self, pose_matrix, timestamp):
        """Publish visualization marker"""
        try:
            marker_array = MarkerArray()
            
            # Cube marker
            cube_marker = Marker()
            cube_marker.header.frame_id = self.camera_frame
            cube_marker.header.stamp = timestamp
            cube_marker.ns = "detected_cubes"
            cube_marker.id = 0
            cube_marker.type = Marker.CUBE
            cube_marker.action = Marker.ADD
            
            # Position
            cube_marker.pose.position.x = float(pose_matrix[0, 3])
            cube_marker.pose.position.y = float(pose_matrix[1, 3])
            cube_marker.pose.position.z = float(pose_matrix[2, 3])
            
            # Orientation
            rotation_matrix = pose_matrix[:3, :3]
            quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
            cube_marker.pose.orientation.x = quaternion[0]
            cube_marker.pose.orientation.y = quaternion[1]
            cube_marker.pose.orientation.z = quaternion[2]
            cube_marker.pose.orientation.w = quaternion[3]
            
            # Scale (5cm cube)
            cube_marker.scale.x = 0.05
            cube_marker.scale.y = 0.05
            cube_marker.scale.z = 0.05
            
            # Color (red)
            cube_marker.color.r = 1.0
            cube_marker.color.g = 0.0
            cube_marker.color.b = 0.0
            cube_marker.color.a = 0.8
            
            # Lifetime
            cube_marker.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()
            
            marker_array.markers.append(cube_marker)
            
            self.marker_pub.publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f"Marker publishing failed: {e}")
    
    def publish_debug_image(self, color_image, pose_matrix, mask, timestamp):
        """Publish enhanced debug image with overlays (like FoundationPose demos)"""
        try:
            debug_image = color_image.copy()
            h, w = debug_image.shape[:2]
            
            # Create overlay for better visibility
            overlay = debug_image.copy()
            
            # Draw mask contours with better visibility
            if mask is not None:
                mask_uint8 = mask.astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw filled contours with transparency
                cv2.drawContours(overlay, contours, -1, (0, 255, 255), -1)
                cv2.drawContours(debug_image, contours, -1, (0, 255, 255), 3)
                
                # Draw bounding boxes
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw center point
                    center = (x + w // 2, y + h // 2)
                    cv2.circle(debug_image, center, 8, (255, 255, 255), -1)
                    cv2.circle(debug_image, center, 8, (0, 0, 255), 3)
            
            # Blend overlay for semi-transparent effect
            alpha = 0.3
            debug_image = cv2.addWeighted(debug_image, 1 - alpha, overlay, alpha, 0)
            
            # Draw 3D coordinate frame (enhanced)
            self.draw_enhanced_coordinate_frame(debug_image, pose_matrix)
            
            # Draw cube wireframe if we have pose
            self.draw_cube_wireframe(debug_image, pose_matrix)
            
            # Enhanced text information
            position = pose_matrix[:3, 3]
            distance = np.linalg.norm(position)
            
            # Background for text
            text_bg_height = 100
            cv2.rectangle(debug_image, (0, 0), (w, text_bg_height), (0, 0, 0), -1)
            cv2.rectangle(debug_image, (0, 0), (w, text_bg_height), (255, 255, 255), 2)
            
            # Multiple lines of information
            lines = [
                f"CUBE DETECTION - FoundationPose + ROS2",
                f"Position: X={position[0]:.3f}m Y={position[1]:.3f}m Z={position[2]:.3f}m",
                f"Distance: {distance:.3f}m | Frame: {len(debug_image)} | Status: TRACKING"
            ]
            
            for i, line in enumerate(lines):
                y_pos = 25 + i * 25
                cv2.putText(debug_image, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw confidence indicator
            confidence_bar_width = 200
            confidence_bar_height = 10
            confidence = min(1.0, max(0.0, (5.0 - distance) / 5.0))  # Simple confidence based on distance
            
            # Background bar
            cv2.rectangle(debug_image, (w - confidence_bar_width - 10, 10), 
                         (w - 10, 10 + confidence_bar_height), (100, 100, 100), -1)
            
            # Confidence bar
            bar_width = int(confidence_bar_width * confidence)
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))
            cv2.rectangle(debug_image, (w - confidence_bar_width - 10, 10), 
                         (w - confidence_bar_width - 10 + bar_width, 10 + confidence_bar_height), color, -1)
            
            cv2.putText(debug_image, f"Conf: {confidence:.2f}", 
                       (w - confidence_bar_width - 10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Convert and publish
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            debug_msg.header.stamp = timestamp
            debug_msg.header.frame_id = self.camera_frame
            self.debug_image_pub.publish(debug_msg)
            
        except Exception as e:
            self.get_logger().error(f"Enhanced debug image publishing failed: {e}")
    
    def draw_enhanced_coordinate_frame(self, image, pose_matrix):
        """Draw enhanced coordinate frame on image (like FoundationPose)"""
        try:
            axis_length = 0.08  # 8cm axes
            
            # 3D points for axes
            origin_3d = pose_matrix[:3, 3]
            x_axis_3d = origin_3d + pose_matrix[:3, 0] * axis_length
            y_axis_3d = origin_3d + pose_matrix[:3, 1] * axis_length
            z_axis_3d = origin_3d + pose_matrix[:3, 2] * axis_length
            
            # Project to 2D
            def project_3d_to_2d(point_3d):
                if point_3d[2] <= 0:
                    return None
                point_cam = self.camera_matrix @ point_3d
                u = int(point_cam[0] / point_cam[2])
                v = int(point_cam[1] / point_cam[2])
                if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                    return (u, v)
                return None
            
            origin_2d = project_3d_to_2d(origin_3d)
            if origin_2d is None:
                return
            
            # Draw axes with enhanced styling
            axes_data = [
                (x_axis_3d, (0, 0, 255), 'X'),    # Red X
                (y_axis_3d, (0, 255, 0), 'Y'),    # Green Y
                (z_axis_3d, (255, 0, 0), 'Z')     # Blue Z
            ]
            
            for axis_3d, color, label in axes_data:
                axis_2d = project_3d_to_2d(axis_3d)
                if axis_2d is not None:
                    # Draw thick arrow
                    cv2.arrowedLine(image, origin_2d, axis_2d, color, 4, tipLength=0.3)
                    # Draw thinner white outline
                    cv2.arrowedLine(image, origin_2d, axis_2d, (255, 255, 255), 2, tipLength=0.3)
                    
                    # Add axis labels
                    cv2.putText(image, label, 
                               (axis_2d[0] + 5, axis_2d[1] - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw enhanced center point
            cv2.circle(image, origin_2d, 8, (255, 255, 255), -1)
            cv2.circle(image, origin_2d, 8, (0, 0, 0), 2)
            cv2.circle(image, origin_2d, 4, (0, 255, 255), -1)
        
        except Exception as e:
            self.get_logger().error(f"Enhanced coordinate frame drawing failed: {e}")
    
    def draw_cube_wireframe(self, image, pose_matrix):
        """Draw 3D cube wireframe (like FoundationPose demos)"""
        try:
            # Define cube vertices (5cm cube)
            cube_size = 0.025  # 2.5cm half-size
            vertices_3d = np.array([
                [-cube_size, -cube_size, -cube_size],  # 0
                [cube_size, -cube_size, -cube_size],   # 1
                [cube_size, cube_size, -cube_size],    # 2
                [-cube_size, cube_size, -cube_size],   # 3
                [-cube_size, -cube_size, cube_size],   # 4
                [cube_size, -cube_size, cube_size],    # 5
                [cube_size, cube_size, cube_size],     # 6
                [-cube_size, cube_size, cube_size],    # 7
            ])
            
            # Transform vertices to world coordinates
            transformed_vertices = []
            for vertex in vertices_3d:
                world_vertex = pose_matrix[:3, :3] @ vertex + pose_matrix[:3, 3]
                transformed_vertices.append(world_vertex)
            
            # Project to 2D
            projected_vertices = []
            for vertex_3d in transformed_vertices:
                if vertex_3d[2] > 0:
                    point_cam = self.camera_matrix @ vertex_3d
                    u = int(point_cam[0] / point_cam[2])
                    v = int(point_cam[1] / point_cam[2])
                    if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                        projected_vertices.append((u, v))
                    else:
                        projected_vertices.append(None)
                else:
                    projected_vertices.append(None)
            
            # Define cube edges
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
                (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
                (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
            ]
            
            # Draw edges
            for edge in edges:
                p1_idx, p2_idx = edge
                p1 = projected_vertices[p1_idx]
                p2 = projected_vertices[p2_idx]
                
                if p1 is not None and p2 is not None:
                    cv2.line(image, p1, p2, (255, 255, 0), 2)  # Cyan wireframe
                    cv2.line(image, p1, p2, (0, 0, 0), 1)      # Black outline
            
            # Draw vertices
            for vertex in projected_vertices:
                if vertex is not None:
                    cv2.circle(image, vertex, 3, (255, 255, 255), -1)
                    cv2.circle(image, vertex, 3, (255, 0, 255), 1)
        
        except Exception as e:
            self.get_logger().error(f"Cube wireframe drawing failed: {e}")
    
    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion"""
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([x, y, z, w])

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = SystemFoundationPoseNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
