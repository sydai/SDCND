# **Project 1: Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I used Gaussian function to smooth the grayscale image. Afterwards, applied Canny edge detection funciton on the smoothed (blurred) image to obtain all the edges of the original image. Since our target is the lanes, I applied a region of interest to the output of last step to obtain only edges of the lane lines. Now came to link the edge points into lines by applying Hough transform with proper parameters such as no_of_votes, min_length_to_connect. After going through Hough transform, there were still line segments. It was needed to fit the line segments which belonged to either of the lanes into a single line (left and right respectively).

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by first dividing all line segments into two groups (left and right lanes) then extrapolating line segments for each group by using a function called extrap(). Inside extrap(), I calculated the slope, intercept and length of each line segment then tried to average all slopes and intercepts based on two criteria: 1. the absolute slope is within the range of +/-0.4 from 0.45, i.e. 0.05~0.95 to eliminate those line segments that is deviating too much from the trend of the lane. 2. assign weights to different line segments based on the length: set weight=3.0 if length larger than the mean length of all line segments otherwise set weight=1.0. After obtaining the average slope and intercept, by further defining the top and bottom limits of the interested region, I'm able to calculate the extreme points of each line and finally plot two lines representing left and right lanes.

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be the averaged slope and intercept not valid when sum of all weights is zero, i.e. division by zero. That is why I chose a larger range of slope i.e. 0.05~0.95 to be involved in calculation of average slope and intercept. But the pool of selected line segments may still include lines that deviate too much from the real slope of the lane thus cause the final fitted line to be inaccurate.

Another shortcoming could be the criteria of assigning more weights to the longer line segments. Hypothetically, if the longer line segments' slopes were far away from the trend of lane line, the final fitted line would be even more inaccurate. In addition, values of the weights will affect the final result and require experiments to determine.

An unclear doubt is how to make the same algorithm, parameters and weights etc apply to all types of lanes (dotted, dashed, curved..) without tuning. The fact is for each of the test images, I tuned the parameters (of Canny, of Hough etc) to make the perfect fitted lines. The problem of no universal algorithm/parameters is rising when running the code on videos: the fitted lines are quite unstable and moving everywhere. 



### 3. Suggest possible improvements to your pipeline

A possible improvement would be to add a check for "division by zero" thus to avoid such problem.

Another potential improvement could be to check slopes by the sign + and - instead of checking the absolute values, i.e. remove "-" slopes for right lane and "+" slopes for left lane from the pool of line segments used to calculate average slope and intercept. 

Other improvements:
1. When testing with images, it's better to use a loop to test multiple images all at one run.
2. Similar with videos
