import cv2
import os
from ultralytics import YOLO
import math
import argparse
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import psutil

# Empty dictionary to store the detections
detections = []

# Empty list to store inference times
inference_times = []

# Empty list to store preprocessing times
postprocessing_times = []

# Empty list to store cpu usage
cpu_usage = []

# Empty list to store memory usage
memory_usage = []

# Empty list to store I/O times
io_times = []

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

    # Initialize the frame count
    frame_count = 0
    
    # Start timer for FPS
    fps_start = cv2.getTickCount()

    # Loop through the video frames
    while ret:
        
        # Get the CPU usage
        cpu_usage.append(psutil.cpu_percent())
        
        # Get the memory usage
        memory_usage.append(psutil.virtual_memory().percent)
        
        # Update the frame count
        frame_count += 1
        
        # Save the frame as an image file
        frame_path = os.path.join(frame_output_folder, f'{frame_count:06}.jpg')
        cv2.imwrite(frame_path, frame)

        # Start timer for I/O
        start_io_time = time.time()
        
        # Read the next frame
        ret, frame = video.read()
        
        # End timer for I/O
        end_io_time = time.time()
        
        # Append the I/O time to the list
        io_times.append((end_io_time - start_io_time) * 1000)
        
        # Start timer for inference
        start_inf_time = time.time()
        
        # Run the model on the frame
        results = model(frame, stream=True, classes=[0])
        
        # End timer for inference
        end_inf_time = time.time()
        
        # Append the preprocessing time to the list
        inference_times.append((end_inf_time - start_inf_time) * 1000)

        # Start timer for postprocessing
        time_pp_start = time.time()
        
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
    
        # If the 'q' key is pressed, break the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        # End timer for inference
        time_pp_end = time.time()

        # Append the inference time to the list
        postprocessing_times.append((time_pp_end - time_pp_start) * 1000)
        
    # Release the video file
    video.release()

    # Write the detections to a file
    with open(os.path.join(detection_output_folder, 'det.txt'), 'w') as f:
        for detection in detections:
            f.write(','.join(map(str, detection)) + '\n')

    # End timer for FPS
    elasped_time = (cv2.getTickCount() - fps_start) / cv2.getTickFrequency()
    
    # Calculate the FPS
    if elasped_time > 0:
        fps = frame_count / elasped_time
        
    return frame_count, fps

# Parse command line arguments

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
    # Parse command line arguments
    args = parse_args()
    
    # Process the video
    total_frames, avg_fps = process_video(args.video_path, args.output_folder)
    
    # Plot the inference times
    plt.plot(range(1, total_frames+1), inference_times)
    plt.xlabel('Frames')
    plt.ylabel('Inference Time (ms)')
    plt.ylim(top=5, bottom=0)
    plt.title('Inference Time')
    plt.figtext(0.5, 0.0, f'Average Inference Time: {sum(inference_times) / len(inference_times):.2f} ms', wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()
    
    # Plot the post-processing times
    plt.plot(range(1, total_frames+1), postprocessing_times)
    plt.xlabel('Frames')
    plt.ylim(top=120, bottom=0)
    plt.ylabel('Post-Processing Time (ms)')
    plt.title('Post-Processing Time')
    plt.figtext(0.5, 0.0, f'Average Post-Processing Time: {sum(postprocessing_times) / len(postprocessing_times):.2f} ms', wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()
    
    # Plot the I/O times
    plt.plot(range(1, total_frames+1), io_times)
    plt.xlabel('Frames')
    plt.ylabel('I/O Time (ms)')
    plt.ylim(top=5, bottom=0)
    plt.title('I/O Time')
    plt.figtext(0.5, 0.0, f'Average I/O Time: {sum(io_times) / len(io_times):.2f} ms', wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()
    
    # Plot the CPU Usage
    plt.plot(range(1, total_frames+1), cpu_usage)
    plt.xlabel('Frames')
    plt.ylabel('CPU Usage (%)')
    plt.ylim(top=100, bottom=0)
    plt.title('CPU Usage')
    plt.figtext(0.5, 0.0, f'Average CPU Usage: {sum(cpu_usage) / len(cpu_usage):.2f} %', wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()
    
    # Plot the Memory Usage
    plt.plot(range(1, total_frames+1), memory_usage)
    plt.xlabel('Frames')
    plt.ylabel('Memory Usage (%)')
    plt.ylim(top=100, bottom=0)
    plt.title('Memory Usage')
    plt.figtext(0.5, 0.0, f'Average Memory Usage: {sum(memory_usage) / len(memory_usage):.2f} %', wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()
    
    print(f'Average FPS: {avg_fps:.2f}')
    
if __name__ == '__main__':
    main()