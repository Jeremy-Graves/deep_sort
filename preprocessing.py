import cv2
import os
from ultralytics import YOLO
import math
import argparse
import tensorflow as tf

def process_video(video_path, output_folder):
    """Process a video file and save the frames and detections.
    
    Parameters
    ----------
    video_path : str
        Path to the video file.
    frame_output_folder : str
        Path to the folder to save the frames.
    detection_output_folder : str
        Path to the folder to save the detections.
    """
    # Load the YOLO model
    model = YOLO('.//Yolo-Weights/yolov8n.pt')
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create the frame output folder if it doesn't exist
    frame_output_folder = os.path.join(output_folder, 'img1')
    os.makedirs(frame_output_folder, exist_ok=True)

    # Create the detections output folder if it doesn't exist
    detection_output_folder = os.path.join(output_folder, 'det')
    os.makedirs(detection_output_folder, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, frame = video.read()
    frame_count = 1

    # Empty dictionary to store the detections
    detections = []

    # Loop through the video frames
    while ret:
        
        # Save the frame as an image file
        frame_path = os.path.join(frame_output_folder, f'{frame_count:06}.jpg')
        cv2.imwrite(frame_path, frame)
        
        # Read the next frame
        ret, frame = video.read()
        
        # Run the model on the frame
        results = model(frame, stream=True, classes=[0])
        
        # Iterate over the results
        for r in results:
            boxes = r.boxes
            
            # Iterate over the boxes in each result
            for box in boxes:
                    
                # Extract the bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  
                w, h = x2-x1, y2-y1
                
                # Confidence
                conf = math.ceil((box.conf[0]*100))/100

                # Class
                cls = box.cls[0]
                
                # Append the detection to the list
                bbox_data = [frame_count,-1, x1, y1, w, h, conf, -1, -1, -1]
                detections.append(bbox_data)
                    
        # Increment the frame count     
        frame_count += 1
        
        # If the 'q' key is pressed, break the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    # Release the video file
    video.release()

    # Write the detections to a file
    with open(os.path.join(detection_output_folder, 'det.txt'), 'w') as f:
        for detection in detections:
            f.write(','.join(map(str, detection)) + '\n')



def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Process a video file and save the frames and detections.")
    
    parser.add_argument(
        "--video_path", help="Path to the video file.", required=True)
    
    parser.add_argument(
        "--output_folder", help="Path to the folder to save the frames.", required=True)
    
    return parser.parse_args()
    
def main():
    args = parse_args()
    process_video(args.video_path, args.output_folder)
    
if __name__ == '__main__':
    main()
