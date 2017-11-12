#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/random_y_train.png "sample image from training set with label 35: Ahead Only"
[image2]: ./dist_y_train.png "distribution of training set"
[image3]: ./dist_y_valid.png "distribution of validation set"
[image4]: ./dist_y_test.png "distribution of test set"
[image5]: ./examples/grayscale.png "grascale of the same train image"
[image6]: ./new_dwn_signs/3_lim60.png "label 3: 60km/h limit"
[image7]: ./new_dwn_signs/34_turnleft.png "label 34: turn left"
[image8]: ./new_dwn_signs/25_roadwork.png "label 25: road work"
[image9]: ./new_dwn_signs/18_caution.png "label 18: caution"
[image10]: ./new_dwn_signs/11_rightofway.png "label 11: right of way"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sydai/SDCND/blob/master/Project2_Traffic_Sign_Classifier/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is len(X_train)
* The size of the validation set is len(X_valid)
* The size of test set is len(X_test)
* The shape of a traffic sign image is X_train[0].shape
* The number of unique classes/labels in the data set is len(set(y_train))

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Below is a plot of a randomly selected image from the training set.


![trainset image][image1]

Here is a bar chart showing how the data from training, validation and test set distributed:
![trainset distribution][image2]

![validset distribution][image3] 

![testset distribution][image4] 

From the distribution, it can be seen that training and validation dataset are distributed in orderly way, i.e. a neighborhood share the same label. While the test set has a random distribution. Therefore, when training the model train set need be shuffled.


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it reduces input channel from 3 to 1 thus decreasing the number of weights parameters needed and shortening the training time

Here is an example of a traffic sign image after grayscaling.

![grayscale trainset image][image5]

As a last step, I normalized the image data because it needs that all data features on the same scale so that all data features weight equally for learning of the model parameters (weights and biases).


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| input 400, outputs 120       	|
| RELU					|												|
| Fully connected		| input 120, outputs 84       	|
| RELU					|												|
| Fully connected		| input 84, outputs 43       	|
| Softmax				|        									|
|	Loss operation					|												|
|	Optimization					|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with 

batch size: 128

number of epochs: 40

learning rate: 0.0009

mean of initial weights: 0

standard deviation of intial weights: 0.1



####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Steps of the iterative process:
1. Inside each epoch
2. Shuffle the training set
3. Divide training set into batches of size batch_size
4. For each batch, feed the training data to the model to update weights and biases
5. At end of each epoch, check the performance of the trained model by feeding the validation set to the model and calculate the accuracy.
During training, I tuned the hyperparameters in a trial and error manner. 


My final model results were:
* validation set accuracy of 0.932 
* test set accuracy of 0.919


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The LeNet model was chosen because it proved to work in the lab. Only the size of input/output depth was changed.
The model worked well so no further change was made. Below questions can be skipped.

* What were some problems with the initial architecture?  
NA
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.   
NA
* Which parameters were tuned? How were they adjusted and why?   

I tuned the hyperparameters in a trial and error manner. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
NA

If a well known architecture was chosen:  NA
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

The 3rd image (label 25: road work) might be difficult to classify because it resembles some of the other traffic signs (e.g. label 31: wild animals crossing).

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn Left      		| Turn Left   									| 
| Road work     			| Wild animals crossing										|
| caution					| caution											|
| 60 km/h	      		| 60 km/h				 				|
| right of way			| right of way      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.919

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 169th cell of the Ipython notebook.

For the first image, the model is absolutely sure that this is a caution sign (probability of 1.0), and the image does contain a caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Caution sign   									| 
| 7.48*e-19     				| Traffic Signals 										|
| 3.87*e-22					| Right of way											|
| 1.97*e-23	      			| Go straight or left					 				|
| 1.44*e-23				    | Pedestrains      							|


For the image that got wrongly classified:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9998         			| Wild animal crossing (wrong class)   									| 
| 1.32*e-4    				| Bicycles crossing										|
| 4.69*e-5				| Go straight or left										|
| 1.04*e-5	      			| Road work (correct class)					 				|
| 3.59*e-7				    | Slippery road      							|




### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


