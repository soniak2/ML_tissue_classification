from PIL import Image
import numpy as np

def prepare_masks(path, classes, tissue_mask):
    threshold = 128
    tissue = Image.open(path + tissue_mask).point(lambda p: p > threshold)

    for cls in classes:
        mask = np.array(Image.open(path + cls))
        result = mask * tissue
        Image.fromarray(result).save(path + 'preprocessed_mask_' + cls)

def count_pixels(path, classes):
    data = []
    counts = []

    for cls in classes:
        mask = Image.open(path + 'preprocessed_mask_' + cls).point(lambda p: p > 128)
        coords = np.argwhere(mask)
        data.append(coords)
        counts.append(coords.shape[0])

    return data, min(counts)