import cv2
import os
from ultralytics import YOLO
import math
import argparse

from preprocessing import process_video
from tools.generate_detections import create_box_encoder
from tools.generate_detections import generate_detections
from deep_sort_app import run
from deep_sort_app import bool_string


def parse_args():
    """Parse command line arguments.
    """
    
    parser = argparse.ArgumentParser(description="Process a video file and save the frames and detections.")
    
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf."
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the video file."
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output folder."
    )
    
    parser.add_argument(
        "--sequence_dir", help="Path to  sequence directory",
        type=str,
        required=True
    )
    parser.add_argument(
        "--ds_output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    
    parser.add_argument(
        "--detection_output", help="Path to the folder to save the detections.",
        required=True
    )
    
    '''parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True
    )'''

    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.5,
        help="Confidence threshold."
    )
    
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int
    )
    
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float
    )
    
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2
    )
    
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None
    )
    
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string
    )
    

    return parser.parse_args()

import os

def main():
    """
    Main function that processes the video, generates detections, and runs deep sort.

    Args:
        None

    Returns:
        None
    """
    # Parse command line arguments
    args = parse_args()

    # Create the sequence path
    sequence_path = os.path.join(args.output, args.sequence_dir)

    # Process the video and save the frames to the sequence path
    process_video(args.video, sequence_path)

    # Create the box encoder
    encoder = create_box_encoder(args.model, batch_size=32)

    # Generate detections and save them to a file
    detection_file = generate_detections(encoder, args.output, args.detection_output)

    # Run deep sort on the sequence with the generated detections
    run(
        sequence_path, detection_file, args.ds_output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display
    )


if __name__ == "__main__":
    main()