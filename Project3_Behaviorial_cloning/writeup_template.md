#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./cnn-architecture-624x890.png "Model Visualization"
[image2]: ./examples/placeholder.png "Center lane driving"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image8]: ./examples/placeholder_small.png "Training and Validation Accuracy"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of 
1. Normalization layer: x/255.0 - 0.5 for deriving mean centered and equal variance input data
2. convolution neural network with 5x5 filter sizes and depths of 24 (model.py lines 119) 
3. convolution neural network with 5x5 filter sizes and depths of 36 (model.py lines 120) 
4. convolution neural network with 5x5 filter sizes and depths of 48 (model.py lines 121) 
5. convolution neural network with 3x3 filter sizes and depths of 64 (model.py lines 122) 
6. convolution neural network with 3x3 filter sizes and depths of 64 (model.py lines 123) 

The model includes RELU activation together with all 5 CNN layers to introduce nonlinearity (code line 119-123), and the data is normalized in the model using a Keras lambda layer (code line 118). 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 65-66, 102-107). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 130).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, clockwise/anti-clockwise direction driving

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to prepare training data, use Keras for model training and finally save the model trained.

My first step was to use a convolution neural network model for feature extraction. After a few convolution neural network layers, I applied 3 fully connected layers for control and steering angles.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I decreased the number of epochs to run and collected new training data. (didn't consider dropout layer)

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track at a sharp right turn and got stucked to the curb after the bridge to improve the driving behavior in these cases, I recorded new training data for similar scenarios and intentionaly made the drive direction from side to center (for the stucked case) and reverse the driving direction for more right turns.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 115-128) consisted of 1 normalizaton layer, 5 convolution layers and 3 fully connected layers.

![https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to keep to the center of the track. These images show what a recovery looks like starting from the side of the road :

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would double the training data and complement for bias of left turns of track one. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had X number of data points. I then preprocessed this data by normalization to equal variance and center meaned data.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by validation accuracy vs training accuracy plot that after 3rd epoch validation accuracy increased. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image8]
