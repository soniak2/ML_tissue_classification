import spectral

def load_metrics(path):
    metrics = spectral.open_image(path)
    print('Loaded metrics:', path)
    return metrics
