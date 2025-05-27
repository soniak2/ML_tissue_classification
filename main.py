from utils.preprocessing import prepare_masks, count_pixels
from utils.features import extract_pixels
from utils.prediction import train_classifier, evaluate_model, predict_tissue
from utils.io import load_metrics
from config import MASKS_PATH, CLASSES, CLASSES_COLORS, TISSUE_MASK
from sklearn.model_selection import train_test_split

# Step 1: Generate or clean binary masks for each tissue class (e.g. cancer, benign tumor, stroma)
# These masks indicate which pixels in the images belong to which class
prepare_masks(MASKS_PATH, CLASSES, TISSUE_MASK)

# Step 2: Count and collect pixel coordinates for each class from the masks
# Also determine the minimum number of pixels available across all classes
indexes, min_pixels = count_pixels(MASKS_PATH, CLASSES)

# Step 3: Load the spectral image using spectral library
# This returns a memory-mapped image object - <class 'spectral.io.bilfile.BilFile'>
metrics = load_metrics("BR2085d_3_5_metrics_new_MNF20.hdr")

# Step 4: Extract spectral data and corresponding labels from the image
# Only min_pixels per class are extracted to ensure class balance
data, labels = extract_pixels(indexes, min_pixels, metrics)

# Step 5: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

# Step 6: Train a machine learning classifier - Random Forest
classifier = train_classifier(X_train, y_train)
evaluate_model(classifier, X_test, y_test)

# Step 8: Use the trained classifier to predict tissue types across the entire image
# Generates an RGB image where each pixel is colored according to its predicted class
predict_tissue(classifier, MASKS_PATH, TISSUE_MASK, metrics, CLASSES_COLORS)