# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "/Users/dai/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/train.p"
validation_file= "/Users/dai/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/valid.p"
testing_file = "/Users/dai/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
#print(X_train[0], len(X_train[0]), y_train)
print(X_train.shape)

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
#import pandas as pd
# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
y_unique=y_train
n_classes = len(set(y_unique))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
#import matplotlib.pyplot as plt
import random
import numpy as np
# Visualizations will be shown in the notebook.
#%matplotlib inline

index=random.randint(0,len(X_train))
image=X_train[index].squeeze()

#plt.figure(figsize=(1,2))
#plt.imshow(image)
print(y_train[index])


#plt.axis([0, len(y_valid), 0, 45])
#plt.plot(y_valid)
#plt.show()


# X_train=(X_train-128)/128
# X_valid=(X_valid-128)/128
# X_test=(X_test-128)/128

def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

def RGB2Gray(dataset):
    for i, img in enumerate(dataset):
        for j, row in enumerate(img):
            for m, pixel in enumerate(row):
                print(weightedAverage(pixel))
                dataset[i][j][m]=weightedAverage(pixel)
                print(dataset[i][j][m])
    return dataset
#print(X_train[0:1])
X_t=RGB2Gray(X_train[0:1])
    #[[[[1,2,3],[3,4,5],[6,7,8]],[[1,2,3],[3,4,5],[6,7,8]],[[1,2,3],[3,4,5],[6,7,8]]],[[[1,2,3],[3,4,5],[6,7,8]],[[1,2,3],[3,4,5],[6,7,8]],[[1,2,3],[3,4,5],[6,7,8]]]])
print ("--------------")
#print(X_t)