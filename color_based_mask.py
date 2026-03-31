import argparse
import os
import sys
import cv2
import numpy as np
from ultralytics import SAM


def detect_colored_objects(img, color_name):
    """Detect objects of specific color and return bounding boxes."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
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
    
    return bboxes   # xywh


def main():
    parser = argparse.ArgumentParser(description="Multi-color cube segmentation with SAM")
    parser.add_argument("--image", type=str, default="/home/student/Desktop/perception/FoundationPose/ros2_ws/src/py_srvcli/resource/color/frame_20250904_191449.png")#"/home/student/Desktop/perception/FoundationPose/demo_data/mustard0/rgb/1581120424100262102.png")
    parser.add_argument("--colors", nargs='+', default=['red', 'blue', 'yellow'], help="Colors to detect")
    parser.add_argument("--model", type=str, default="mobile_sam.pt", help="Colors to detect")
    parser.add_argument("--out-dir", type=str, default="/home/student/Desktop/perception/FoundationPose/demo_data/cube/masks", help="Output directory for masks")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load image and SAM model
    img = cv2.imread(args.image)    # H x W x 3 (BGR)
    seg_model = SAM("sam2.1_l.pt")#args.model)
    seg_model.to("cuda")
    
    all_masks = []
    mask_info = []
    
    # Process each color
    for color in args.colors:
        print(f"Detecting {color} objects...")
        bboxes = detect_colored_objects(img, color)
        
        if not bboxes:
            print(f"No {color} objects found.")
            continue
            
        # Run SAM on all bboxes for this color
        results = seg_model.predict(source=args.image, bboxes=bboxes, device="cuda")
        
        if results and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            
            # Save individual masks for this color
            for i, mask in enumerate(masks):
                m_bin = mask.astype(np.uint8) * 255
                filename = f"{color}_cube_{i:02d}.png"
                filepath = os.path.join(args.out_dir, filename)
                cv2.imwrite(filepath, m_bin)
                
                all_masks.append(m_bin)
                mask_info.append(f"{color}_{i}")
                print(f"Saved: {filename}")
    
    # Create combined visualization
    if all_masks:
        combined = np.zeros_like(all_masks[0])
        for i, mask in enumerate(all_masks):
            combined = np.maximum(combined, mask // len(all_masks) * (i + 1))
        
        cv2.imwrite(os.path.join(args.out_dir, "all_cubes_combined.png"), combined)
        print(f"Total objects detected: {len(all_masks)}")
        print(f"Saved combined mask and {len(all_masks)} individual masks to: {args.out_dir}")
    else:
        print("No objects detected.")


if __name__ == "__main__":
    main()


# python SAM_Ultranalytics.py --image /home/student/Desktop/perception/FoundationPose/demo_data/mustard0/rgb/1581120424100262102.png --model mobile_sam.pt --out mustard_mask.png --device 0