## Advanced Lane Finding Writeup

**Advanced Lane Finding Project**

The goals / steps of this project were the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/chessboard.png "Undistorted"
[image2]: ./examples/undistorted.png "Road Transformed"
[image3]: ./examples/binary_combo_examples.png "Binary Example"
[image4]: ./examples/warpExample.png "Warp Example"
[image5]: ./examples/FitLines.png "Fit Visual"
[image6]: ./examples/FinalOutput.png "Output"
[video1]: ./project_video_with_lane.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### The rubric points are addressed below individually; I describe how I addressed each point in my implementation for each point.

Because the initial submited was rejected due to some frames of the video with failed lane drawing, I have also added the following items:

1) Added HSV thresholding along with full image and HSL thresholding for white and yellow specifically, and combined that with limited sobel and red channel thresholds to get better initial lane identification.  See processimage().

2) Added, and later removed, polygon shape comparison between current and prior frames with cv2.matchshape().  While this handled some outliers, it had a tendency to get stuck if a bad prediction went on for more than two frames.

3) Replaced the cv2.matchchape() with an array tracking the prior 10 frames x fit polynomial lines.  If the left and right xfit arrays varied too widely from the prior 10 frames, I would not include that prediction.  Actual lane findnig was then based on the mean of either the mean of the current frame and the prior 9, or the mean of the prior 10.  See slidingwindowsearch().

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.   

This writeup meets this criteria directly.  Below I will address all the subsequent requirements, and include examples and code references for each.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "Project4.ipynb". 

Initially I set up the objp arrays to hold the discovered inner corners in a chessboard.  This will allow me to determine the distortion introduced by the camera lens used through the exercise.  The chessboard pattern used for this calibration is a 6*9 black and white board printed on a sheet of paper and taped to a wall.  The images of the chessboard can then be fed through a series of calibration functions provided by the cv2 library in order to calculate the distortion aspects of the lens.

I start off by collecting all the chessboard photos to be used from the \camera_cal\ directory.  Looping through each image in that directory, I use the cv2.findChessboardCorners() and a grayscale copy of the image to locate the inner corners of the chessboard, which are added to the objpoints array as the x,y,z coordinates of each point (where z=0).  I then draw lines connecting each point/inner corner, though I'm only displaying the very last instance for the sake of expediency.

I use the collected points to then execute cv2.callibrateCamera(), which returns the calibration matrix I'll need to undistort the source images from the driving videos later.  To verify that the undistortion is working, I use the calibration matrix for this lens to display an undistorted chessboard sample.  

I then take the outermost 4 corners returned by the corner-finding function, and perspective warp the chessboard image so that it is square to the image plane.  By mapping the four chessboard corners to a rectangle in an output image, I can then use the cv2.getPerspectiveTransform() function to take the undistorted image from the last step, and warp it to be square to the viewer's perspective.  This will be useful later when I'm warping driving images in order to more easily detect line curvature.

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The distortion correction can then be applied to any image from this lens at this zoom level.  Since the camera in question is using a fixed lens, then any image it produces can be corrected with this process.

To demonstrate this, the below road image is the output of just the distortion correction step.  The correction is present, though not extreme.  This is because the lens is not an extremely wide-angled lens like a fish-eye, and the longer distance of the subject from the camera reduces the apparent distortion; the effect is much more noticeable in the close-up chess board images.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In the processimage() function under the heading "Next, load the road images and manipulate the data to better see lane lines", I take in an individual image and a pair of threshold min and max values.  Using this image, I first take the R channel from the RGB format an store it.  I then convert the image to HLS format using the cv2.cvtColor() function.  I then break out the color channels into individual arrays, and use them to calculate different binary versions of the image for easier manipulation later.  I first calculate sobelx with cv2.Sobel() using the l_channel, and convert that output to a binary image by checking if each pixel is within the threshold range passed into the function.  I use the s_channel to calculate an s_binary in the same way.  I calculate an h_binary from the h_channel, using a very low threshold to try and handle darker images as well as low-contrast areas.  Finally, I combine these binary images into a single binary output by flipping each pixel in a black image to white if at least one of the three binary images above has a white pixel in that position.

I had originally relied on the R_channel I isolated first to help pull out the lane lines, as it generally produced a very high-contrast line in all test images.  However, in the project video the R-channel failed to differentiate the lefthand lane line at all in areas with light colored concrete, swamping the binary image with noise.  For now, I have removed the R channel, and replaced it with the h_binary; while there is considerable added noise in the top half of the image from h_binary, the top half is ignored during lane finding so it should not impact the result.  There are still some issues in the lane finding during certain conditions; when a dark car reflecting lane lines in its door panels pass, and when a tree shades the road during a transition from dark to light concrete, but the results are better than with the R-channel alone.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called warpperspective(), which appears in the 6th code cell in the notebook, under the comment/heading ##Perspective Transform.  The warpperspective() function takes as inputs an image 'imagein', and uses preset source and destination points to undo the vanishing point perspective inherent to all images of receding landscapes.  By defining a polygon and then transforming it to a rectangle, I can force the further points to appear at the top of an image, mimicking a top-down view of the roadway.  I chose the hardcode the source and destination points in the following manner:

```python
    src_vertices = np.float32(
        [[200, image.shape[0]],
        [575, int(image.shape[0]*.64)],         
        [702, int(image.shape[0]*.64)], 
        [1110,  image.shape[0]]])
    
    dst_vertices = np.float32(
        [[400, 720],
        [400, 0], 
        [900, 0], 
        [900, 720]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 400,720       | 
| 575, 460      | 400, 0        |
| 702, 460      | 900, 0        |
| 1110, 720     | 900, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image called "topdown".

![alt text][image4]

The  warpperspective() function returns, in addition to the Topdown image, the matrix needed to unwarp the image, "resultMinv".  This will be used later when drawing the identified lane area to the original, unwarped image of the lane.

I tested this process with all the available test images, plotting each with various combinations of 

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the next code cells, I implemented the slidingwindowsearch() code provided in the lesson.  While it did identify the lane lines with fair accuracy, then second slidingwindowsearch() code block, also from the lesson, does a more thorough job.  This sliding windowsearch() function includes code to fit polynomials to the curves of the discovered lines using np.polyfit().

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Once slidingwindowsearch() had identified the lanes and the np.polyfit() function used to fit polynomial functions to their curves, we encounter a problem.  The polynomials are based off of the camera image, and not the real world.  By using the provided formulas for the known width and length of US highway lanes, were can correct for this, and re-fit new polynomials. This is done in the find_curvature() function.  

In the position_to_center() function, I also adjust for image to real-world scaling, after finding the difference between the center of the image, and the center of the lane.  This is returned as the offset between the car position and the actual lane center.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Once found, I then draw the lane overlay in the function drawlane(), by taking the discovered lane shape, creating a polygon of that shape and filling it with green, then using the "resultMinv" calculated in the warpperspective() function to unwarp the polygon to fit the original image's perspective.  Combining these images with an alpha to the green lane provides an output image matching the original, with the current lane area highlighted.

Lastly, I added text to the image's top left corner, displaying the curvature and off-center position of the car, already calculated.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Only then did I have all the pieces I needed to build an actual pipeline.  This block of code, named find_lane() and found under the comment/heading "Organize the above into a single function" takes in a single color image, undistorts it, processes it to produce binary output, performs a perspective warp on it, uses slidingindowsearch() to identify lane lines, calculates the lane curve and center, then overlays the lane area onto the original image, along with the measured curve and position values.

After testing this on a few test images, I applied it to the project video.

Here's a [link to my video result](./project_video_with_lane.mp4)

I also applied the process to the challenge_video.mp4 file, with results displayed in the third from final cell in the notebook.  The performance on this challenge video were better than my initial attempts, particularly when the HOV lane contained two colors of cement, but was still sub-par.  During the shadow under the bridge, and for a while after, the process failed almost completely.  I would like to have more time to continue working on the pipeline, but there is much left to cover in this Term, and not many days to cover it!

Finally, I also tested the harder_challenge_video.mp4 file, with little expectation that the output would be effective.  The result ranged from suprisingly passible to dangerously terrible, mostly due to the process' inability to detect the right-hand lane line.  This was suprising to me, given that it appeared obvious.  Instead the job tended to favor the edge of the grass just beyond the right lane line instead of the line itself.  During a very strong right hand corner, the predictions turned into a spagetti mess as the camera lost sight of the right lane line, and the left lane line existed as a curved diagonal nearly horizontal across the screen.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

By switching from a strictly RGB.grayscale approach to one using different color spaces, the lane finding process improved significantly over the initial attempt at the beginning of this course.  My suggestion in that initial writeup of processing each RBG color channel individually was actually borne out by this project, but it was also short-sighted in not considering HSL as a colorspace to use.

The lens distortion correction used is familiar to me from Adobe Lightroom and correcting for different lenses on SLR cameras, however it was very cool to see exactly how that correction can be done programmatically.  The chessboard correction is a simple an powerful method to correct both distortion and perspective warping in one process.

Similarly, I had ideas of how we might improve lane finding and curvature calculations, but I had not thought of perspective warping to artificially create an overhead lane view.  This turned out to be much more effective than I had expected, and with the cv2 functions available, performing this feat was not too complicated.

What I do feel is nearly magic is the sliding window search process.  While I have read the code and the descriptions provided, I'm still flummoxed by the details of how this function actually works; I will continue studying it in order to improve my understanding as we approach the car identification project.

Merging the Behavioral Cloning process with this process would be an interesting one.  If we could teach a neural network not just to clone behavior, but to clone object identification behavior, we could start to generate a rudimentary conceptualization within an A.I.  In addition to rote learning ("when o your this, turn this far that way"), we could also begin to train an A.I. to identify things within the images it processes, and use modern-day decision trees to "think ahead", and plan expectations of future states based on what those objects are most likely to do.

Handling the extremly strong curve of the harder challenge video will require altering the assuming that the lines we are looking for will be vertical, and would benefit from more heavily weighting prior lane identification from the past few frames of video to predict where the line is likely to be next.  By following the left line as it curves further and further right, the process will be less likely to lose it completely.