# Advanced Lane Finding


#### Advanced Lane Finding Project

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

[image1]: ./output_images/demo_calibration.jpg "Calibration"
[image2]: ./output_images/demo_distortion_correction.jpg "Undistort"
[image3]: ./output_images/demo_gray_sobel.jpg "Grayscale_threshold"
[image4]: ./output_images/demo_sat_sobel.jpg "HLS_S_threshold"
[image5]: ./output_images/demo_sobel_threshold.jpg "Threshold"
[image6]: ./output_images/demo_perspective.jpg "Perspective"
[image7]: ./output_images/demo_perspective2.jpg "Perspective2"
[image8]: ./output_images/demo.jpg "Demo"
[image9]: ./output_images/demo_sliding_window.gif "Sliding_window"
[image10]: ./output_images/demo_mimic_points.png "Mimic"
[image11]: ./output_images/demo_targeted_search.gif "Targeted_search"
[image12]: ./output_images/demo_smoothing_ewma_filter.gif "EWMA"
[image13]: ./output_images/demo_result.png "Result"

#### Submission

My project includes the following files:
* `run.py` is the main source containing the pipeline
* `camera.py` containing camera calibration, distortion correction and perspective transform
* `imgroc.py` containing thresholding based on color thresholds and Sobel gradients
* `writeup.md` summarizing the results

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 42-73 in `camera.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The function `undistort()` (in lines 75-77 in `camera.py`) uses the previously computed camera matrix and distortion coefficients to compensate for camera distortions. The following test image shows only minor differences, most noticeable at car hood.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is contained in lines 1-84 in `imgproc.py`. I used a combination of color transforms and gradient thresholds to generate a binary image. The function `threshold()` in lines 63-84 summarizes the thresholding steps. The base images for gradient calculations were grayscale image and HLS S-channel. I applied Sobel operators in both X and Y directions, then combined the results of directions and then the results of different base images.

![alt text][image3]
![alt text][image4]

```python
def threshold(img, show_results=False):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,2]

    gray_gradx = sobel_thresh(gray, orient='x', thresh=(20, 255), sobel_kernel=5)
    gray_grady = sobel_thresh(gray, orient='y', thresh=(20, 255), sobel_kernel=5)
    # gray_mag = magnitude_thresh(gray, thresh=(20, 255), sobel_kernel=3)
    # gray_dir = direction_thresh(gray, thresh=(0.7, 1.3), sobel_kernel=15)
    sat_gradx = sobel_thresh(sat, orient='x', thresh=(20, 255), sobel_kernel=5)
    sat_grady = sobel_thresh(sat, orient='y', thresh=(20, 255), sobel_kernel=5)

    combined = np.zeros_like(gray)
    combined[((gray_gradx == 1) & (gray_grady == 1)) | ((sat_gradx == 1) & (sat_grady == 1))] = 1

    return combined
```
Here's an example of my output for this step (results for test_images are saved in `output_images` folder).
![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for perspective transform resides the file `camera.py` in lines 79-94. The related functions are  called `calc_perspective()`, `warp()` and `unwarp()`. The `calc_perspective()` function uses values defined in the Camera() constructor. The source  points were calculated based on `test_images/test5.jpg`. The destination points were tweaked until the result was realistic enough.

```python
        # values for perspective transform
        self.x = (163, 540, 768, 1139)
        self.y = (720, 489)
        self.nx = (651-100, 651+100)
        self.ny = (720, 489)
        ...

    def calc_perspective(self):
        # compute the perspective transform, M, given source and destination points
        x, y, nx, ny = self.x, self.y, self.nx, self.ny
        src = np.float32([[x[0], y[0]], [x[1], y[1]], [x[2], y[1]], [x[3], y[0]]])
        dst = np.float32([[nx[0], ny[0]], [nx[0], ny[1]], [nx[1], ny[1]], [nx[1], ny[0]]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        return self.M, self.Minv

    def warp(self, img):
        # warp an image to bird's eye view using the perspective transform self.M
        return cv2.warpPerspective(img, self.M, img.shape[:2][::-1], flags=cv2.INTER_LINEAR)

    def unwarp(self, img):
        # unwarp an image from bird's eye viewusing the perspective transform self.M
        return cv2.warpPerspective(img, self.Minv, img.shape[:2][::-1], flags=cv2.INTER_LINEAR)

```

This resulted in the following source and destination points:

| Source   | Destination|
| :-------:| :---------:|
| 163, 720 | 551, 720   |
| 540, 489 | 551, 489   |
| 768, 489 | 751, 489   |
| 1139, 720| 751, 720   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Results for test_images are saved in `output_images` folder.

![alt text][image6]
![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After applying undistortion, thresholding and perspective transform to a road image
I used sliding window approach or targeted search to identify pixels that belong to lines on the warped image. Then I fit second order polynomials to both lane line pixels using  `numpy.polyfit()`. To avoid frame to frame jumps of detected lines I calculated moving average on the polynomial coefficients.

![alt text][image8]

##### Sliding window approach

This was performed as blind search when previous line positions could not be used for calculation. Line base coordinates were calculated using histogram, adding up the pixel values along each column in the image. Median was either at half width or at the center of previous left/right bases. Histogram peaks are selected as left/right bases, but only within a clipping distance of 150 pixels from median. In the histogram above for example the adjacent lane line has higher value, but it is over the clipping distance, so the smaller closer value will be selected as right base. From the base points I use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame. I used 12 windows with margin of 35 pixels. I set limits that subsequent windows must not change rapidly, more than 50 pixels sideways.

![alt text][image9]

When minimum number of 50 pixels was not found to recenter window, I added pixels into window centers. These mimic points were added to empty windows to avoid curly polyfits at top/bottom. This means that empty window centers took part in polynomial fitting as we can see in the image below.

![alt text][image10]


##### Targeted search

When lines were previously correctly detected, search in a 30 pixels margin around previous line positions should be just enough. The moving average filtered best polyfit was used as line position. Two sanity checks could approve a successful targeted search. One that the left/right line curve radiuses should not differ too much, their ratio is expected to be smaller than 2. Also top and bottom lane width should be similar, their difference is expected to be less than 10 percent. When one of these checks failed, sliding windows was the fallback mechanism to find lane lines.

![alt text][image11]


##### Exponentially weighed moving average

There is short description of EWMA filtering [here](https://www.mcgurrin.info/robots/?p=154). It seemed like a good idea to apply this to polynomial coefficients and line curve radius. The image below show a situation when polyfit is changing rapidly and EWMA performs the smoothing (yellow line is actual, purple is smoothed).

![alt text][image12]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The function `get_curvature()` and `get_lane_position()` (in lines 312-334 in `run.py`) provides these values. Both functions uses EWMA filtered polyfit. Radius of curvature uses the equation from lecture, lane pixel values are converted from pixel space to real world space. The values `ym_per_pix` and `ym_per_pix` are manually calculated based on undistorted and warped `test5.jpg`. For the position of the vehicle, I assumed the camera is mounted at the center of the car.

```python
def get_curvature(img, update=True):
    # real world meter/pixel is measured on test5.jpg
    ym_per_pix = 3.05/45    # typical lane line is 3.05 meters (10 feet)
    xm_per_pix = 3.66/195   # typical lane width is 3.66 meters (12 feet)
    left_fit = np.polyfit(left.pixels[0]*ym_per_pix, left.pixels[1]*xm_per_pix, 2)
    right_fit = np.polyfit(right.pixels[0]*ym_per_pix, right.pixels[1]*xm_per_pix, 2)
    y_level = img.shape[0]*ym_per_pix
    left.curve = ((1 + (2*left_fit[0]*y_level + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right.curve = ((1 + (2*right_fit[0]*y_level + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    if update:
        left.update_curve(left.curve)
        right.update_curve(right.curve)
    return (left.best_curve + right.best_curve)/2

def get_lane_position(img):
    left_fit = left.best_fit
    right_fit = right.best_fit
    xcar = img.shape[1]//2
    h = int(img.shape[0]*0.8)
    xleft = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
    xright = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
    # typical lane width is 3.66 meters (12 feet)
    return (xcar - (xleft+xright)//2) / (xright-xleft) * 3.66
```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 77 through 93 in my code in `run.py` in the function `process_image()`.  Here is an example of my result on a test image:

![alt text][image13]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

##### General issues

If I were to make my pipeline more robust I would tune thresholding a bit to get more pixels of lane lines even if it means more 'noise'. With more advanced sanity checks it seems better to detect more than have nothing from distant lines. I'd like to build more on previous line information and leave mimic points out of the game.

##### Challenge #1

This video requires more correct image thresholding. Color thresholding combined with gradient thresholding might do the trick. This is quite an experiment with lots of values to tune. Recently I encountered `ipywidgets.interact`, very useful. Had I known it earlier, challenge #1 would have been more fun to solve.

##### Challenge #2

This is where `sliding_window()` and `targeted_search()` functions would need some enhancement. I implemented early stopping in sliding window approach at some point that did not advance to upper windows once sanity checks fail. It was promising but not enough. Now I ran out of time.
