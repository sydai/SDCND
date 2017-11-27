import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2

# Get size of calibration image
img = cv2.imread('camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# 1. Camera calibration
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
   
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
   
# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')
   
# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
       
    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
   
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #write_name = 'corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
        #cv2.imshow('img', img)
        cv2.waitKey(500)
#print(corners[0,0])
#print(np.shape(corners))
#print(np.shape(imgpoints))
cv2.destroyAllWindows()
   
import pickle
# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
   
# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "output_images/cal_dist_pickle.p", "wb" ) )
     
       
# 2. Distortion correction
# Test undistortion on an image
img = cv2.imread('test_images/test2.jpg')
# undistort test image
undist = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('output_images/test2_undist.jpg',undist)
# Visualize undistortion   
undist = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB) #convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(undist)
ax2.set_title('Undistorted Image', fontsize=30)
pyplot.savefig('output_images/test2_vs_undistort.png')
  
def select_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([20,60,60])
    upper = np.array([38,174, 250])
    mask = cv2.inRange(hsv, lower, upper)
    
    return mask

def select_white(image):
    lower = np.array([202,202,202])
    upper = np.array([255,255,255])
    mask = cv2.inRange(image, lower, upper)
    
    return mask

img = mpimg.imread('output_images/test2_undist.jpg')
def doubleThreshold(image):
    yellow = select_yellow(image)
    white = select_white(image)
    
    combined_binary = np.zeros_like(yellow)
    combined_binary[(yellow >= 1) | (white >= 1)] = 255
    
    return combined_binary  
threshed=doubleThreshold(img)
cv2.imwrite('output_images/test2_undist_thresh.jpg',threshed)

# 3. Color/gradient threshold
# Read in an image and grayscale it
# img = mpimg.imread('output_images/cal_undist.jpg')
# def doubleThreshold(img):
#     # Convert to HLS color space and separate the S channel
#     # Note: img is the undistorted image
#     hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     s_channel = hls[:,:,2]
#       
#     # Grayscale image
#     # NOTE: we already saw that standard grayscaling lost color information for the lane lines
#     # Explore gradients in other colors spaces / color channels to see what might work better
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#       
#     # Sobel x
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
#     abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
#     scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
#       
#     # Threshold x gradient
#     thresh_min = 30
#     thresh_max = 100
#     sxbinary = np.zeros_like(scaled_sobel)
#     sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
#       
#     # Threshold color channel
#     s_thresh_min = 170
#     s_thresh_max = 255
#     s_binary = np.zeros_like(s_channel)
#     s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1     
#     # Combine the two binary thresholds
#     combined_binary = np.zeros_like(sxbinary)
#     combined_binary[(s_binary == 1) | (sxbinary == 1)] = 255
#     print(np.shape(combined_binary))
#     return combined_binary
# 
# threshed=doubleThreshold(img)
# cv2.imwrite('output_images/cal_undist_thresh.jpg',threshed)
# #Plotting thresholded images
# fig = plt.figure()
# plt.title('Combined S channel and gradient thresholds')
# plt.imshow(threshed, cmap='gray')
# plt.show()
# pyplot.savefig('output_images/cal_undist_thresh.png')
 
 
# 4. Perspective Transform
# Read image from last step
img = mpimg.imread('output_images/undist_thresh.jpg')
def perspectiveTrans(img):
#     offset=50
#     up=np.int(0.65*img_size[1]) #vertical height
#     down=np.int(img_size[1])
#     leftup=np.int(0.45*img_size[0]) #horizontal
#     leftdown=np.int(0.2*img_size[0])
#     rightup=np.int(0.6*img_size[0])
#     rightdown=np.int(0.85*img_size[0])
#     hist_leftup = np.sum(img[up:(up+offset),leftup-offset:leftup], axis=0)
#     hist_leftdown = np.sum(img[down-offset:down,leftdown:leftdown+offset], axis=0)
#     hist_rightup = np.sum(img[up:(up+offset),rightup:rightup+offset], axis=0)
#     hist_rightdown = np.sum(img[down-offset:down,rightdown-offset:rightdown], axis=0)
#      
#     leftup = np.argmax(hist_leftup)+leftup-offset #argmax returns index of max value
#     leftdown = np.argmax(hist_leftdown)+leftdown-offset
#     rightup = np.argmax(hist_rightup)+rightup-offset
#     rightdown = np.argmax(hist_rightdown)+rightdown-offset
#     #
#     src = np.float32([[leftup,up], [rightup,up], [leftdown,down], [rightdown,down]])
#     dst = np.float32([[leftdown, 0], [rightdown,0], [leftdown, down], [rightdown,down]])
    #src = np.float32([[599,447], [680,447], [227,704], [1065,704]])
    
    src = np.float32([[545, 460],
                    [735, 460],
                    [1280, 700],
                    [0, 700]])

    dst = np.float32([[0, 0],
                     [1280, 0],
                     [1280, 720],
                     [0, 720]])
    
#     print(src)
#     print(dst)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    MInv = cv2.getPerspectiveTransform(dst,src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    org = cv2.warpPerspective(warped, MInv, img_size)
    return warped, org, MInv

warped, org, MInv = perspectiveTrans(img)
cv2.imwrite('output_images/test2_undist_thresh_transform.jpg',warped)
# cv2.imwrite('output_images/undist_thresh_invtrans.jpg',org)



# 5. Detect lane lines
# Read in an image and grayscale it
img = mpimg.imread('output_images/undist_thresh_transform.jpg')
ploty = np.linspace(0, img_size[1]-1, img_size[1] ) #args: (start,stop,elements)
def slide_window(img):
    # detect lane lines by sliding windows
    binary_warped=img
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[np.int(img_size[1]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    #out_img = np.dstack((binary_warped, binary_warped, binary_warped))#*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint]) #argmax returns index of max value
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img_size[1]/nwindows) # y vertical height
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero() #nonzero() returns tuple of arrays. e.g.coord:(3,2) (1,7),(8,6) -> ([3,1,8],[2,7,6])
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base #current is center of window in x axis
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img_size[1] - (window+1)*window_height
        win_y_high = img_size[1] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
    #     cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
    #     (0,255,0), 2) 
    #     cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
    #     (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2) #lefty in 1st place, similar to x axis as in ax^2+bx+c
    right_fit = np.polyfit(righty, rightx, 2) #returns polynomial coeffients a,b,c (xx_fit[0],[1],[2])
    
    # Generate x and y values for plotting
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2] # all pairs of (ploty,left_fitx) form a polynomial line (power 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] # as later plotted line in yellow color
    return leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx

leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx = slide_window(img)

def searchAfter(img,undist,left_fit,right_fit):
    binary_warped=img
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin)))  # use +/- margin criteria for nonzero indexes be included in lane lines
    
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] ) 
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]  # all x values along lane line for each y value
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Project those lines onto the original image as follows:
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    lane_mask=np.copy(color_warp)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # mark lane points with red color on left lane and blue on right lane
    lane_mask[lefty, leftx] = [255, 0, 0]
    lane_mask[righty, rightx] = [0, 0, 255]
    color_warp = cv2.addWeighted(color_warp, 1, lane_mask, 1, 0)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, MInv, img_size) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)
    # Color in left and right line pixels
    #result=cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
    return leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx, result

leftx, lefty, rightx, righty, left_fit, right_fit,left_fitx, right_fitx, result=searchAfter(img,undist,left_fit,right_fit)
result=cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
# cv2.imwrite('output_images/shaded_lane.jpg',result)


#6. Determine the lane curvature
def laneCurvature(leftx,lefty,rightx,righty,left_fit,right_fit):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    left_basex = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_basex = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    
    road_center=(left_basex+right_basex)*0.5
    vehicle_center=img_size[0]*0.5
    abs_diff=np.abs(road_center-vehicle_center)*xm_per_pix
    if vehicle_center<road_center:
        shift='left'
    else:
        shift='right'
    return left_curverad,right_curverad,shift,abs_diff

left_curverad,right_curverad,shift,abs_diff=laneCurvature(leftx,lefty,rightx,righty,left_fit,right_fit)
# print(left_curverad, 'm,', right_curverad, 'm', '.vehicle is', abs_diff,'m', shift,'of center')


class Line():
    def __init__(self,recent_xfitted,recent_fit,current_fit,radius,allx,ally):
        # was the line detected in the last iteration?
        self.detected = True  
        # x values of the last n fits of the line
        self.recent_xfitted = recent_xfitted 
        #average x values of the fitted line over the last n iterations
        self.bestx = np.mean(self.recent_xfitted,axis=0)  
        # last n fits of the line
        self.recent_fit = recent_fit
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.mean(self.recent_fit,axis=0)  
        #polynomial coefficients for the most recent fit
        self.current_fit = current_fit 
        #radius of curvature of the line in some units
        self.radius_of_curvature = radius
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.abs(self.current_fit-self.recent_fit[-1])
        #x values for detected line pixels
        self.allx = allx  
        #y values for detected line pixels
        self.ally = ally
    def sanityCheck(self):
        if (self.radius_of_curvature/1000<0.4) or (self.radius_of_curvature/1000>3):
            self.detected=False
        last_fit=self.recent_fit[-1]
        if np.any(self.diffs>[0.01*last_fit[0], 0.01*last_fit[1], 0.01*last_fit[2]]):
            self.detected=False
        
      
def process_image(img):
    global frame
    global lastFrame
    
    frame+=1
    # 2. Undistortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
     
    # 3. gradient & color thresholding
    threshed=doubleThreshold(undist)
     
    # 4. Perspective transform
    warped, org, MInv = perspectiveTrans(threshed)
    
    # 5. Detect lane lines
    # first frame use slide window to locate lanes
    if frame==1:
        leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx = slide_window(warped)
        left_curverad,right_curverad,shift,abs_diff=laneCurvature(leftx,lefty,rightx,righty,left_fit,right_fit)
    else: # from 2nd frame onwards, search around last detected lanes for current lanes
        left_fit, right_fit = lastFrame["left_fit"],lastFrame["right_fit"]
        leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx, result=searchAfter(warped,undist,left_fit,right_fit)
        left_curverad,right_curverad,shift,abs_diff=laneCurvature(leftx,lefty,rightx,righty,left_fit,right_fit)
    
    if frame<=avg_n:
        recent_left_fitx.append(left_fitx)
        recent_left_fit.append(left_fit)
        recent_right_fitx.append(right_fitx)
        recent_right_fit.append(right_fit)
        #left
        line_l_current=Line(recent_left_fitx,recent_left_fit,left_fit,left_curverad,leftx,lefty)
        #right
        line_r_current=Line(recent_right_fitx,recent_right_fit,right_fit,right_curverad,rightx,righty)
        
    if frame>avg_n:     
        recent_left_fitx.pop(0)
        recent_left_fitx.append(left_fitx)
        recent_left_fit.pop(0)
        recent_left_fit.append(left_fit)
        recent_right_fitx.pop(0)
        recent_right_fitx.append(right_fitx)
        recent_right_fit.pop(0)
        recent_right_fit.append(right_fit)
        #left
        line_l_current=Line(recent_left_fitx,recent_left_fit,left_fit,left_curverad,leftx,lefty)
        line_l_current.sanityCheck()
        #right
        line_r_current=Line(recent_right_fitx,recent_right_fit,right_fit,right_curverad,rightx,righty)
        line_r_current.sanityCheck()
        if line_r_current.detected=='False' or line_l_current.detected=='False':
            print("sanity check failed")
            leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx = slide_window(warped)   
        
        
    #print('frame',frame,':',line_l_current.recent_fit,line_l_current.best_fit) 
    #leftx, lefty, rightx, righty, left_fit, right_fit,left_fitx,right_fitx, result=searchAfter(warped,undist,left_fit,right_fit)    
    leftx, lefty, rightx, righty, left_fit, right_fit,left_fitx,right_fitx, result=searchAfter(warped,undist,line_l_current.best_fit,line_r_current.best_fit)
    
    # 6. Determine the lane curvature and center shift
    left_curverad,right_curverad,shift,abs_diff=laneCurvature(leftx,lefty,rightx,righty,line_l_current.best_fit,line_r_current.best_fit)
    
    # For use in next frame
    lastFrame={"leftx":leftx,"lefty":lefty, "rightx":rightx, "righty":righty,"left_fit":left_fit,"right_fit":right_fit,
          "left_fitx":left_fitx, "right_fitx":right_fitx, "left_curve":left_curverad,"right_curve":right_curverad}
    
    text1= "left curvature %f m, right curvature %f m" % (left_curverad,right_curverad)
    text2= "vehicle is %f m %s of road center" % (abs_diff, shift)
    font = cv2.FONT_HERSHEY_SIMPLEX
    result=cv2.putText(result,text1,(100,100), font, 1,(0,0,255),2,cv2.LINE_AA)
    result=cv2.putText(result,text2,(100,200), font, 1,(0,0,255),2,cv2.LINE_AA)
    
    return result

avg_n = 20
recent_left_fitx=[] 
recent_left_fit =[]
recent_right_fitx=[] 
recent_right_fit =[]
frame=0
lastFrame={}

from moviepy.editor import VideoFileClip
clip1 = VideoFileClip("project_video.mp4")#.subclip(40,48)
out_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
video_output = 'output_images/out_project_video.mp4'
out_clip.write_videofile(video_output, audio=False)

