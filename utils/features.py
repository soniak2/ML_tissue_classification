import numpy as np

def extract_pixels(indexes, min_count, metrics):
    bands = metrics.shape[2]
    total = len(indexes) * int(min_count)

    data = np.zeros((total, bands), dtype=np.float32)
    labels = np.zeros(total, dtype=np.int32)

    idx = 0
    for label, coords in enumerate(indexes):
        selected = np.random.choice(coords.shape[0], int(min_count), replace=False)
        for i in selected:
            x, y = coords[i]
            data[idx, :] = metrics.read_pixel(x, y)
            labels[idx] = label
            idx += 1
    return data, labels
