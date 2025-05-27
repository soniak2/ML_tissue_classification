from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time

def train_classifier(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=50,
                                 criterion='gini', 
                                 max_depth=None, 
                                 min_samples_split=2, 
                                 min_samples_leaf=1, 
                                 random_state=42, 
                                 verbose=1
                                 )
    clf.fit(X_train, y_train)
    return clf


'''
def train_classifier(X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    clf = RandomForestClassifier(random_state=42, verbose=1)
    grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_
'''



def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print("Classification report:\n", classification_report(y_test, y_pred))



def predict_tissue(clf, path, tissue_mask, metrics, class_colors):
    start = time.time()
    mask = Image.open(path + tissue_mask).point(lambda p: p > 128)
    mask = np.array(mask)
    indexes = np.argwhere(mask)

    metrics = metrics.load()
    data = metrics[indexes[:, 0], indexes[:, 1], :]
    nan_mask = np.isnan(data).any(axis=1)

    valid_data = data[~nan_mask]
    valid_coords = indexes[~nan_mask]

    predictions = clf.predict(valid_data)
    image = Image.new("RGB", (metrics.shape[1], metrics.shape[0]))

    for (x, y), label in zip(valid_coords, predictions):
        image.putpixel((y, x), class_colors[label])

    # Add legend
    draw = ImageDraw.Draw(image)
    legend_labels = ['cancer', 'benign tumor', 'stroma']

    font_size = 48
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
        print('Error - font')

    padding = 20
    rect_size = font_size 

    box_height = (rect_size + padding) * len(class_colors)
    box_width = 460
    x_offset = image.width - box_width - 20
    y_offset = 20

    draw.rectangle(
        [x_offset - 10, y_offset - 10, x_offset + box_width, y_offset + box_height],
        fill=(0, 0, 0, 180)  
    )

    for i, (color, label) in enumerate(zip(class_colors, legend_labels)):
        y = y_offset + i * (rect_size + padding)
        draw.rectangle([x_offset, y, x_offset + rect_size, y + rect_size], fill=color)
        draw.text((x_offset + rect_size + 15, y), label, fill=(255, 255, 255), font=font)

    # Save image
    image.save("tissue_prediction_cancer_benigntumor_stroma.jpg")
    print("Prediction done in {:.2f}s".format(time.time() - start))




#Best parameters: {'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}