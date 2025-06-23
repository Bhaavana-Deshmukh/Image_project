import os
import sys
from src.exception import CustomException
from src.logger import logging
from cellpose import models
import cv2, imageio, pandas as pd
from skimage.measure import regionprops
import numpy as np


SUPPORTED_FORMATS = ('.avi', '.mp4', '.mov', '.mkv', '.tif', '.tiff')
video_path = "E:/projects/cell_classification_project/input/my_cells.avi"

def validate_input_path(video_path: str) -> None:
    """
    Checks whether the given path exists and is in a supported video/image format.
    """
    if not os.path.exists(video_path):
        logger.error(f"File not found: {video_path}")
        raise CustomException(f"File not found at: {video_path}", sys)
    
    if not video_path.lower().endswith(SUPPORTED_FORMATS):
        logger.error(f"Unsupported file format: {video_path}")
        raise CustomException(
            f"Unsupported file format. Supported formats: {SUPPORTED_FORMATS}", sys
        )
    
    logger.info(f"Validated input path: {video_path}")

def load_video_reader(video_path: str):
    try:
        reader = imageio.get_reader(video_path)
        total_frames = reader.count_frames()
        fps = reader.get_meta_data().get("fps", 10)
        logger.info(f"Loaded video: {video_path} | Frames: {total_frames}, FPS: {fps}")
        return reader, total_frames, fps
    

    except Exception as e:
        logger.exception(f"Error occurred while loading video from path: {video_path}")
        raise CustomException(e, sys)

