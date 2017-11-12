import os
import csv
import cv2
import numpy as np

lines = []
car_images=[]
steering_angles=[]
with open('/home/carnd/training_morex3/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
##        steering_center=float(line[3])
##        correction=0.2
##        steering_left=steering_center+correction
##        steering_right=steering_center-correction
##        path='/home/carnd/training_more/IMG/'
##        file1=line[0].split('/')[-1]
##        file2=line[1].split('/')[-1]
##        file3=line[2].split('/')[-1]
##        img_center=cv2.imread(path+file1)
##        img_left=cv2.imread(path+file2)
##        img_right=cv2.imread(path+file3)
##        car_images.extend([img_center,img_left,img_right])
##        steering_angles.extend([steering_center,steering_left,steering_right])
##print(file1)
##print(np.shape(img_center))
##print(np.shape(car_images))
##print(np.shape(steering_angles))


from sklearn.model_selection import train_test_split
train_samples, validation_samples=train_test_split(lines,test_size=0.2)

import sklearn
path='/home/carnd/training_morex3/IMG/'
correction=0.1
def generator(samples,batch_size=32):
    num_samples = len(samples)
    while 1: #Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples=samples[offset:offset+batch_size]

            images=[]
            angles=[]
            images_center_only=[]
            angles_center_only=[]
            for batch_sample in batch_samples:
                name1=path+batch_sample[0].split('/')[-1]
                name2=path+batch_sample[1].split('/')[-1]
                name3=path+batch_sample[2].split('/')[-1]
                img_center=cv2.imread(name1)
                img_left=cv2.imread(name2)
                img_right=cv2.imread(name3)
                angle_center=float(batch_sample[3])
                angle_left=angle_center+correction
                angle_right=angle_center-correction
                images.extend([img_center,img_left,img_right])
                angles.extend([angle_center,angle_left,angle_right])
                images_center_only.append(img_center)
                angles_center_only.append(angle_center)
            #X_train=np.array(images)
            #y_train=np.array(angles)
            X_train=np.array(images_center_only)
            y_train=np.array(angles_center_only)
            yield sklearn.utils.shuffle(X_train,y_train)
            
train_generator=generator(train_samples,batch_size=32)
validation_generator=generator(validation_samples,batch_size=32)

ch,row,col=3,65,320 #Trimmed image format

##n=0
##images=[]
##measurements=[]
##for line in lines:
##    for i in range(3):
##        source_path=line[i]
##        filename=source_path.split('/')[-1]
##        current_path='/home/carnd/training_more/IMG/'+filename
##        image=cv2.imread(current_path)
##        images.append(image)
##        n+=1
##        measurement=float(line[3])
##        measurements.append(measurement)

#print(n)
#print(np.shape(image))
#print(np.shape(images))

##augmented_images, augmented_measurements=[],[]
##for image,measurement in zip(images,measurements):
##    augmented_images.append(image)
##    augmented_measurements.append(measurement)
##    augmented_images.append(cv2.flip(image,1))
##    augmented_measurements.append(measurement*-1.0)
##flip_image=np.flip(image,1)
#print(np.shape(flip_image))
#print(np.shape(augmented_images))

#X_train=np.array(images)
#y_train=np.array(measurements)
#X_train=np.array(augmented_images)
#y_train=np.array(augmented_measurements)
#X_train=np.array(car_images)
#y_train=np.array(steering_angles)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#model from NVIDIA
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0)))) #remove upper 70 pixel rows and lower 25 pixel rows
#model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(ch,row,col),output_shape=(ch,row,col)))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu")) #kernel size: 5x5
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu")) #output filters: 36
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu")) #strides:(2,2)
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True,nb_epoch=5, verbose=1)
history_object = model.fit_generator(train_generator,samples_per_epoch=len(train_samples),
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples),
                                     nb_epoch=3,verbose=1)

model.save('model.h5')


from keras.models import Model
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('agg')
plt.switch_backend('agg')

### print the keys contained in the history object
print(history_object.history.keys())
print(history_object.history['loss'])
print(history_object.history['val_loss'])
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('stats.png')
plt.show()



