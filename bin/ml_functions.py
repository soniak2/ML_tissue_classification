
from PIL import Image
import numpy as np
from pandas import isnull
import time

def count_pixels(files_path, classes):

    array_size=np.zeros(len(classes))
    threshold = 128
    data=[]
    for i in range(len(classes)):
        # Read mask
        mask = Image.open(files_path+ 't_' + classes[i])
        # Project image data to boolean type based on the threshold
        mask = mask.point(lambda pixel: pixel > threshold)
        true_coordinates = np.argwhere(mask)
        data.append(true_coordinates)
        array_size[i]= true_coordinates.shape[0]  
        
    M=min(array_size)
    
    return data, M
        

def mask_multiplication(files_path, classes, tissue_mask):
    
    threshold = 128
    tissue = Image.open(files_path+tissue_mask)
    tissue = tissue.point(lambda pixel: pixel > threshold)
    
    for i in range(len(classes)):
        mask = np.array(Image.open(files_path+classes[i]))
        
        mask = mask * tissue
        
        mask_image = Image.fromarray(mask)

        # Save the image as a BMP file
        mask_image.save(files_path + "t_" + classes[i])
    
    return
        
        
def read_pixels(indexes,M,metrics):
     
    num_bands = metrics.shape[2]
     
    #Read data
    data = np.zeros((len(indexes)*int(M),num_bands), dtype=np.float32)
    labels = np.zeros((len(indexes)*int(M)), dtype=np.float32)
    k = 0
    for i in range(len(indexes)):
        print(f"Loading class number {i}...")
        coordinates = np.random.randint(0,(indexes[i].shape)[0],int(M))
        
        for j in range(int(M)):
            X,Y = indexes[i][coordinates[j]]
            t = (metrics.read_pixel(X,Y)).reshape((1,123))
            data[k,:] = t
            labels[k] = i
            k=k+1
             
    return data, labels

def predict_tissue(rf_classifier, masks_path, tissue_mask, metrics, classes_colors):
    
    #Read in tissue mask mask
    mask = Image.open(masks_path + tissue_mask)
    
    threshold = 128
    
    #COnvert into bool
    mask = mask.point(lambda pixel: pixel > threshold)
    
    #Find position of non-zero elements
    indexes = np.argwhere(mask)
    
    #total number of samples
    M = indexes.shape[0]
    
    num_rows = metrics.shape[0]
    num_columns = metrics.shape[1]
    
    #Allocate memory for the prediction image
    image = Image.new("RGB", (num_rows,num_columns))
    print("Prediction...")
    for i in range(M):

        X,Y = indexes[i]
        data = (metrics.read_pixel(X,Y)).reshape((1,123))
        
        if np.any(isnull(data)):
            print("Warning: Pixel skiepped due to NaN values.")
        else:
            label_pred = rf_classifier.predict(data)
            
            if label_pred==0:
                image.putpixel((X,Y), classes_colors[0])
            elif label_pred==1:
                image.putpixel((X,Y), classes_colors[1])  
            else:
                image.putpixel((X,Y), classes_colors[2])
 
        
    output_image = "tissue_prediction.jpg"
    image.save(output_image)
    print("Done.")
    
    return

      
import numpy as np
from PIL import Image

def predict_tissue_fast(rf_classifier, masks_path, tissue_mask, metrics, classes_colors):
    start_time = time.time()
    # Read in tissue mask
    mask = Image.open(masks_path + tissue_mask)
    
    threshold = 128
    
    # Convert into bool
    mask = mask.point(lambda pixel: pixel > threshold)
    
    # Find position of non-zero elements
    indexes = np.argwhere(mask)
    
    # Total number of samples
    M = indexes.shape[0]
    
    num_rows, num_columns = metrics.shape[:2]  # assuming metrics is a 3D array
    
    # Allocate memory for the prediction image
    image = Image.new("RGB", (num_columns, num_rows))
    print("Prediction...")
    
    # Extracting X, Y coordinates
    X, Y = indexes[:, 0], indexes[:, 1]
    
    print(X, Y)
    
    X = np.array(X)
    Y = np.array(Y)
    
    print(X, Y)
    
    data = metrics[X, Y, :]  # Extract relevant data for prediction
    
    # Check for NaN values
    nan_indices = np.isnan(data).any(axis=1)
    print(f"Warning: {nan_indices.sum()} Pixels skipped due to NaN values.")
    
    # Predict labels for non-NaN data
    valid_data = data[~nan_indices]
    valid_indexes = indexes[~nan_indices]
    valid_predictions = rf_classifier.predict(valid_data)
    
    # Set colors based on predictions
    for (X, Y), label_pred in zip(valid_indexes, valid_predictions):
        image.putpixel((Y, X), classes_colors[label_pred])
        
    output_image = "tissue_prediction_fast.jpg"
    image.save(output_image)
    print("Done.")

    end_time = time.time()
    print(end_time - start_time)
    
    return





