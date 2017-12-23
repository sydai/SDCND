**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/colorConv_Hog.png
[image3]: ./examples/window_img.png
[image4]: ./examples/test1_sub_sample_2.png
[image5]: ./examples/test1_CarPos_and_HeatMap_2.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 25 through 42 of the file called `pipeline.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images (9000 for each category).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and decided to use `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features and color features (color binning features and color histogram features) of 6000 vehicles' and 8800 non-vehicles' images with size 64x64.The images come from datasets downloaded from the course site plus cutouts from the project video.
The code for this step is contained in lines 259 through 326 of the file called `pipeline.py`.  

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions by window size 64x64, 96x96, 128x128 (realized by setting scale to 1.0, 1.5, 2.0) with overlap of 0,875% (with cells_per_step equal to 1) all over the image and came up with this:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 1.0, 1.5 and 2.0 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided an accurate result of vehicle detection.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./out_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

I implemented collections.deque for saving a history of 8 heatmaps and then calculated a moving average of heatmaps for thresholding. In this way, many false positives were eleminated and resulted in more stable bounding boxes.

Another method to reduce false positives was using cutouts of the output video where false detection happened in previous training and duplicated these cutouts to certain multifold amount then included in current training data.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is one example and its corresponding heatmap:

![alt text][image5]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The performance of the vehicle detection on project video was initially not so good. Even though the classifier accuracy on the test datasets reached 99.5% (later figured out because of overfitting), when giving a test image to the find_cars() function the detection of vehicles failed most of the times in the video. Due to time constraint, the previous final output video actually missed detections of many vehicles and output many false positives. After implementing the advice from first submission, the final output video showed correct detection of other vehicles with only a few false positives occasionally. For the 4th submission after implementing a moving average method, the false positives were effectively reduced. But somehow the true detections seem not as good as before due to raising of the threshold value (for rejecting false positives).

