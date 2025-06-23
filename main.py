import os, imageio, cv2, pandas as pd
import numpy as np
from skimage.measure import label
from cell_classifier.segmentation import load_model, segment_frame
from cell_classifier.classification import classify_objects
from cell_classifier.visualizer import draw_centroids, write_frame, create_mask_frame

from cell_classifier.data_ingestion import load_video_reader

# --- CONFIGURATION ---
video_path = r"Input/Hela_CM30.avi"
output_folder = "output"
mask_video_path = os.path.join(output_folder, "masks_only.avi")
overlay_video_path = os.path.join(output_folder, "mask_overlay.avi")
table_path = os.path.join(output_folder, "cell_counts.csv")

fixation_frame = 10  # Frame when fixation happens
area_thresh = 50
circ_thresh = 0.5

# --- Load video and model ---
reader = imageio.get_reader(video_path)
fps = reader.get_meta_data()['fps']
frame_size = None
model = load_model(gpu=False)

# --- Prepare video writers ---
mask_writer = None
overlay_writer = None
results = []

for frame_index, frame in enumerate(reader):
    if frame_size is None:
        h, w = frame.shape[:2]
        frame_size = (w, h)
        mask_writer = cv2.VideoWriter(mask_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size, isColor=False)
        overlay_writer = cv2.VideoWriter(overlay_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size, isColor=True)

    # Convert to grayscale if needed
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame

    # Segment and label masks
    masks = segment_frame(model, gray)
    labeled = label(masks)

    # Classification
    counts, centroids = classify_objects(labeled, frame_index, fixation_frame, area_thresh, circ_thresh)

    # Save classification result for table
    result_row = {"Frame": frame_index} | counts
    results.append(result_row)

    # Write mask-only video
    mask_img = (labeled > 0).astype(np.uint8) * 255
    mask_writer.write(mask_img)

    # Overlay mask on original frame
    overlay = frame.copy()
    overlay[masks > 0] = [0, 255, 0]  # green mask
    overlay_writer.write(overlay)

# --- Cleanup ---
reader.close()
mask_writer.release()
overlay_writer.release()

# --- Save table ---
df = pd.DataFrame(results)
df.to_csv(table_path, index=False)

print("✅ All files saved to 'output/' folder.")