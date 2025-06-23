
from cellpose import models

def load_model(gpu=False):
    return models.Cellpose(gpu=gpu, model_type='cyto')

def segment_frame(model, image):
    masks, flows, styles, diams = model.eval(image, channels=[0, 0])
    return masks
    