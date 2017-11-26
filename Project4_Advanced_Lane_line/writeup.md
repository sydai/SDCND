
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/cal_vs_undistort.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/vs_undistort.png "Test1 undistorted"
[image4]: ./output_images/test1_undist_thresh.png "Test1 tresholded"
[image5]: ./output_images/undist_thresh_transform.jpg "Test1 warped"
[image6]: ./output_images/windowed_detection.png "Test1 window sliding"
[image7]: ./output_images/shaded_lane.jpg "Test1 reverse transformed with lane marked"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 14 through 56 of the file caleed 'pipeline.py'.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

By using the "dist" distortion coefficients and "mtx" camera matrix obtained from the calibration as described above. The result of distortion correction is like below:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 80 through 111 in `pipeline.py`).  Here's an example of my output for this step for the same test1 image.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspectiveTrans()`, which appears in lines 126 through 155 in the file `pipeline.py` (./pipeline.py).  The `perspectiveTrans()` function takes as inputs an image (`img`), and inside calculated source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
    offset=50
    up=np.int(0.65*img_size[1]) #vertical height
    down=np.int(img_size[1])
    leftup=np.int(0.45*img_size[0]) #horizontal
    leftdown=np.int(0.2*img_size[0])
    rightup=np.int(0.6*img_size[0])
    rightdown=np.int(0.85*img_size[0])
    hist_leftup = np.sum(img[up:(up+offset),leftup-offset:leftup], axis=0)
    hist_leftdown = np.sum(img[down-offset:down,leftdown:leftdown+offset], axis=0)
    hist_rightup = np.sum(img[up:(up+offset),rightup:rightup+offset], axis=0)
    hist_rightdown = np.sum(img[down-offset:down,rightdown-offset:rightdown], axis=0)
     
    leftup = np.argmax(hist_leftup)+leftup-offset #argmax returns index of max value
    leftdown = np.argmax(hist_leftdown)+leftdown-offset
    rightup = np.argmax(hist_rightup)+rightup-offset
    rightdown = np.argmax(hist_rightdown)+rightdown-offset
    #
    src = np.float32([[leftup,up], [rightup,up], [leftdown,down], [rightdown,down]])
    dst = np.float32([[leftdown, 0], [rightdown,0], [leftdown, down], [rightdown,down]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 526, 468      | 245, 0        | 
| 767, 468      | 1087, 0      |
| 245, 720     | 245, 720      |
| 1087, 720      | 1087, 720        |

I verified that my perspective transform was working as expected by drawing the warped Test1 image to verify that the lines appear relatively parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did window sliding in lines 167 through 245 by first deriving histograms horizontally on the undistorted, thresholded and warped image then use max histogram index as base lane point. Secondly, starting from the base points, search left/right lane points by sliding windows up the image with the criteria of shifting window center if min number of points in the window is not satisfied. Finally identify all points on the left/right lanes with which fit lane lines with a 2nd order polynomial kinda like this:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 308 through 333 in my code in `pipeline.py`. By using the 2nd order polynomial line fitted from all lane points and the formula for calculating curvature, I calculated the curvature at the base of lane lines given the meter per pixel condition so that curvature value is valid (in unit 'meter')

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 249 through 300 in my code in `pipeline.py` in the function `searchAfter()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/out_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

My output video was roughly stable across frames in marking lane lines when they are very clear. I used sanity check for curvature and polynomial coefficient difference across neighboring frames: if curvature<400m or >3000m or if coeff difference>1%, sanity check is failed. When sanity check fails, window sliding will be performed for current frame instead of searching based on lane lines detected in last frame (by function searchAfter()). Refer to lines 339 through 268 for the class 'Line' description and lines 395 through 422.

I also used smoothing in the same class 'Line', passed fitted curves of last n frames into the current left/right lane instance of class 'Line', averaged the last n fitted curves, output as current lane marking.

The pipeline failed when there's a car on the neighboring lane. It seemed when a car appeared on the next lane, lane line detection went to the front tire of the other car instead of own lane line. The problem may be in warping process that src points were not exactly on the lane lines. 

Another problem is the green lane marking still wobbled a bit despite the smoothing process. 
