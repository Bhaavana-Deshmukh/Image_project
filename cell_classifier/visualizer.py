import cv2
import numpy as np

def draw_centroids(frame, centroids, color=(255, 0, 0)):
    for x, y in centroids:
        cv2.circle(frame, (x, y), 5, color, -1)
    return frame

def write_frame(writer, frame):
    writer.append_data(frame)

def create_mask_frame(mask):
    logging.info("Mask frame created")
    return (mask > 0).astype(np.uint8) * 255