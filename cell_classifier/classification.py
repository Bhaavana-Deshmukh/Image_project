import os
import math
from skimage.measure import regionprops
from cell_classifier.utils import calculate_circularity

def calculate_circularity(region):
    return 4 * math.pi * region.area / (region.perimeter ** 2 + 1e-6)

def classify_objects(masks, frame_index, fixation_frame, area_thresh=50, circ_thresh=0.5):
    props = regionprops(masks)
    counts = {"Live": 0, "Fixed": 0, "Dead": 0, "Fragment": 0}
    centroids = []

    for r in props:
        area = r.area
        circ = calculate_circularity(r)
        y, x = r.centroid
        centroids.append((int(x), int(y)))

        if area < area_thresh or circ < circ_thresh:
            counts["Fragment"] += 1
        else:
            if frame_index < fixation_frame:
                counts["Live"] += 1
            elif frame_index == fixation_frame:
                counts["Fixed"] += 1
            else:
                counts["Dead"] += 1

    return counts, centroids
