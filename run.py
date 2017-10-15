#!/usr/bin/env python3

import os, sys
import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from PIL import ImageFont, ImageDraw, Image

from camera import Camera
from imgproc import *

# define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # starting point for the most recent line
        self.base = None
        # detected line pixels
        self.pixels = None
        # sliding line rectangles
        self.rectangles = None
        # polynomial coefficients for the most recent fit
        self.fit = [np.array([False])]
        # EWMA filtered polynomial coefficients
        # [https://www.mcgurrin.info/robots/?p=154]
        self.best_fit = None
        # line curvature
        self.curve = None
        # EWMA filtered lane curvature
        self.best_curve = None

    def update_fit(self, fit):
        # update EWMA filter of polynomial coefficients
        if self.best_fit is not None:
            self.best_fit = fit + 0.8 * (self.best_fit - fit)
        else:
            self.best_fit = fit

    def update_curve(self, curve):
        # update EWMA filter of line curvature
        if self.best_curve is not None:
            self.best_curve = curve + 0.8 * (self.curve - curve)
        else:
            self.best_curve = curve
        return self.best_curve

left = Line()
right = Line()


######################
#      PIPELINE
######################

def process_image(img):
    # correct camera lens distortions
    img = cam.undistort(img)
    # apply threshold based on Sobel gradients of grayscale and HLS S-channel
    timg = threshold(img, show_results=False)
    # crop region of interest, not that important because warping basically removes unneeded area
    #timg = crop_roi(timg)
    # warp image to bird's eye perspective
    wimg = cam.warp(timg)

    if not left.detected or not right.detected:
        # blind search when previous line positions cannot be used for calculation
        sliding_window(wimg)
    else:
        # we are confident that lines were previously correctly detected
        # search in a margin around previous line positions
        targeted_search(wimg)

    # create an image to decorate
    temp = np.zeros_like(wimg).astype(np.uint8)
    temp = np.dstack((temp, temp, temp))

    if left.fit is not None and right.fit is not None:
        # custom decoration when fit is available
        temp = decorate_lane(temp)
        #temp = decorate_rectangles(temp)
        #temp = decorate_search_window(temp)
        #temp = decorate_pixels(temp)
        #temp = decorate_polynomial(temp, False)
        temp = decorate_polynomial(temp)

    # warp back to original image space using inverse perspective matrix (Minv)
    unwarped = cam.unwarp(temp)
    # combine the result with the original image
    out = cv2.addWeighted(img, 1.0, unwarped, 0.8, 0)

    # output in warped space, just for diag
    #out = cv2.addWeighted(cv2.cvtColor(wimg*255,cv2.COLOR_GRAY2RGB), 0.3, temp, 0.8, 0)

    out = show_text_info(out)

    return out


######################
#   HELPER FUNCTIONS
######################

def sliding_window(binary_warped):
    # average lane width is about 190 pixels in warped image
    # so on bottom of first frame lines shall not be farther away from center than
    clip_dist = 150
    # subsequent windows must not change rapidly
    max_win_dist = 50
    # mimic points diameter in center of empty window to avoid curly polyfits
    d = 4

    # choose the number of sliding windows
    nwindows = 12
    # set the width of the windows +/- margin
    margin = 35
    # set minimum number of pixels found to recenter window
    minpix = 50

    (height, width) = binary_warped.shape
    # take a histogram of the bottom 1/4 part of the image
    histogram = np.sum(binary_warped[int(height*0.75):,:], axis=0)
    # find the peak of the left and right halves of the histogram
    # these will be the starting point for the left and right lines
    if left.base is None and right.base is None:
        midpoint = np.int(histogram.shape[0] // 2)
    else:
        midpoint = (left.base + right.base) // 2
    # clear histogram far from lane midpoint
    histogram[:midpoint-clip_dist] = 0
    histogram[midpoint+clip_dist:] = 0

    left.base = np.argmax(histogram[:midpoint])
    right.base = np.argmax(histogram[midpoint:]) + midpoint

    # set height of windows
    window_height = np.int(height//nwindows)
    # identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # current positions to be updated for each window
    (leftx_current, rightx_current) = (left.base, right.base)
    # create empty lists to receive left and right lane pixel indices
    (left_lane_inds, right_lane_inds) = ([], [])
    # create empty lists for window rectangles
    (left.rectangles, right.rectangles) = ([], [])
    # placeholder for mimic points
    (mimic_xl, mimic_yl, mimic_xr, mimic_yr) = ([], [], [], [])

    # step through the windows one by one
    for window in range(nwindows):
        # identify window boundaries in x and y (and right and left)
        win_y_low = height - (window+1)*window_height
        win_y_high = height - window*window_height
        win_xleft_low = max(leftx_current - margin, 0)
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = min(rightx_current + margin, width-1)
        # draw the windows on the visualization image
        left.rectangles.append((win_xleft_low, win_y_low, win_xleft_high, win_y_high))
        right.rectangles.append((win_xright_low, win_y_low, win_xright_high, win_y_high))
        # identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # if you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_candidate = np.int(np.mean(nonzerox[good_left_inds]))
            # sanity check on new window centers
            if abs(leftx_current - leftx_candidate) < max_win_dist:
                leftx_current = leftx_candidate
        else:
            # place some mimic points to window center to avoid curly polyfits
            # if window is nearly empty, its center will be still subject to polyfit
            wx = (win_xleft_low + win_xleft_high) // 2
            wy = (win_y_low + win_y_high) // 2
            (mimic_xl, mimic_yl) = mimic_generator(mimic_xl, mimic_yl, wx, wy, d)

        if len(good_right_inds) > minpix:
            rightx_candidate = np.int(np.mean(nonzerox[good_right_inds]))
            # sanity check on new window centers
            if abs(rightx_current - rightx_candidate) < max_win_dist:
                rightx_current = rightx_candidate
        else:
            # place some mimic points to window center to avoid curly polyfits
            # if window is nearly empty, its center will be still subject to polyfit
            wx = (win_xright_low + win_xright_high) // 2
            wy = (win_y_low + win_y_high) // 2
            (mimic_xr, mimic_yr) = mimic_generator(mimic_xr, mimic_yr, wx, wy, d)

    # concatenate the arrays of indices
    left_lane_inds  = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # extract left and right line pixel positions
    (leftx, lefty)   = (nonzerox[left_lane_inds], nonzeroy[left_lane_inds])
    (rightx, righty) = (nonzerox[right_lane_inds], nonzeroy[right_lane_inds])

    # include mimic points at empty window centers into polyfit
    if len(mimic_xl) > 0:
        leftx = np.concatenate((leftx, np.array(mimic_xl)))
        lefty = np.concatenate((lefty, np.array(mimic_yl)))

    if len(mimic_xr) > 0:
        rightx = np.concatenate((rightx, np.array(mimic_xr)))
        righty = np.concatenate((righty, np.array(mimic_yr)))

    # fit a second order polynomial to each
    left.fit, right.fit = (None, None)
    left.pixels, right.pixels = (None, None)
    if len(leftx) != 0:
        left.fit = np.polyfit(lefty, leftx, 2)
        left.update_fit(left.fit)
        left.pixels = (lefty, leftx)
        left.detected = True
    else:
        left.detected = False

    if len(rightx) != 0:
        right.fit = np.polyfit(righty, rightx, 2)
        right.update_fit(right.fit)
        right.pixels = (righty, rightx)
        right.detected = True
    else:
        right.detected = False

def targeted_search(binary_warped):
    # search in a margin around the previous line position
    # set the width of the windows +/- margin
    margin = 30
    # set minimum number of pixels found to perform polyfit
    minpix = 400
    # max curve difference ratio
    max_curve_diff = 2
    # max lane width difference ratio
    max_width_diff = 1.1

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left.best_fit[0]*(nonzeroy**2) + left.best_fit[1]*nonzeroy + left.best_fit[2] - margin)) &
         (nonzerox < (left.best_fit[0]*(nonzeroy**2) + left.best_fit[1]*nonzeroy + left.best_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right.best_fit[0]*(nonzeroy**2) + right.best_fit[1]*nonzeroy + right.best_fit[2] - margin)) &
         (nonzerox < (right.best_fit[0]*(nonzeroy**2) + right.best_fit[1]*nonzeroy + right.best_fit[2] + margin)))

    # again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # check if we have enough pixels within margin to fit polynom
    results_acceptable = True
    if len(lefty) > minpix and len(righty) > minpix:
        # fit a second order polynomial to each
        left.fit = np.polyfit(lefty, leftx, 2)
        right.fit = np.polyfit(righty, rightx, 2)
        left.pixels = (lefty, leftx)
        right.pixels = (righty, rightx)
        # sanity check on curves
        get_curvature(binary_warped, False)
        ratio = max((left.curve,right.curve))/min((left.curve,right.curve))
        if ratio > max_curve_diff:
            results_acceptable = False
        # sanity check on lane width
        # top and botom lane width should not differ too much
        ytop = int(binary_warped.shape[0]*0.05)
        ybottom = int(binary_warped.shape[0]*0.95)
        xleft_top = left.fit[0]*ytop**2 + left.fit[1]*ytop + left.fit[2]
        xright_top = right.fit[0]*ytop**2 + right.fit[1]*ytop + right.fit[2]
        xleft_bottom = left.fit[0]*ybottom**2 + left.fit[1]*ybottom + left.fit[2]
        xright_bottom = right.fit[0]*ybottom**2 + right.fit[1]*ybottom + right.fit[2]
        width_top = xright_top - xleft_top
        width_bottom = xright_bottom - xleft_bottom
        ratio = max((width_top,width_bottom))/min((width_top,width_bottom))
        if ratio > max_width_diff:
            results_acceptable = False
    else:
        results_acceptable = False

    if results_acceptable:
        # update lines
        left.update_fit(left.fit)
        left.rectangles = None
        right.update_fit(right.fit)
        right.rectangles = None
    else:
        # fallback to search window
        left.detected = False
        right.detected = False
        sliding_window(binary_warped)

def mimic_generator(mimic_x, mimic_y, wx, wy, d):
    # returns coordinates of a rectangle centered at (wx,wy)
    # rectangle diameter is d
    # rectangle coordinates are accumulated in mimic_x, mimic_y
    for x in range(wx-d,wx+d+1):
        for y in range(wy-d,wy+d+1):
            mimic_x.append(x)
            mimic_y.append(y)
    return mimic_x, mimic_y

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

def show_text_info(img):
    curve = get_curvature(img)
    offset = get_lane_position(img)
    text_curve = 'Radius of curvature: {:1.0f}m'.format(curve)
    text_pos1 = '{0:>1.3f}m'.format(abs(offset))
    text_pos2 = ' left of center' if offset<0 else ' right of center'
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text_curve, (10, 36), font, 1.2, (240,240,200), 2, cv2.LINE_AA)
    cv2.putText(img, text_pos1 + text_pos2, (10, 80), font, 1.2, (240,240,200), 2, cv2.LINE_AA)
    return img

def decorate_rectangles(img):
    if left.rectangles is not None and right.rectangles is not None:
        for r in (left.rectangles + right.rectangles):
            cv2.rectangle(img,(r[0],r[1]),(r[2],r[3]), (0,0,255), 2)
            cv2.rectangle(img,(r[0],r[1]),(r[2],r[3]), (0,0,255), 2)
    return img

def decorate_search_window(img):
    if left.rectangles is None and right.rectangles is None:
        margin = 35
        y_min = 0
        ploty = np.linspace(y_min, img.shape[0]-1, img.shape[0]-y_min)
        left_fitx = left.best_fit[0]*ploty**2 + left.best_fit[1]*ploty + left.best_fit[2]
        right_fitx = right.best_fit[0]*ploty**2 + right.best_fit[1]*ploty + right.best_fit[2]
        # generate a polygon to illustrate the search window area
        # and recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        # draw the lane onto the warped blank image
        cv2.fillPoly(img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(img, np.int_([right_line_pts]), (0,255, 0))
    return img

def decorate_polynomial(img, filtered=True):
    left_fit = left.best_fit if filtered else left.fit
    right_fit = right.best_fit if filtered else right.fit
    y_min = 0
    ploty = np.linspace(y_min, img.shape[0]-1, img.shape[0]-y_min)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    radius = 5 if filtered else 3
    color = (180,20,255) if filtered else (255,200,0)
    for xl, xr, y in zip(left_fitx, right_fitx, ploty):
        cv2.circle(img, (int(xl),int(y)), radius, color, -1)
        cv2.circle(img, (int(xr),int(y)), radius, color, -1)
    return img

def decorate_lane(img):
    y_min = 0
    ploty = np.linspace(y_min, img.shape[0]-1, img.shape[0]-y_min)
    left_fitx = left.best_fit[0]*ploty**2 + left.best_fit[1]*ploty + left.best_fit[2]
    right_fitx = right.best_fit[0]*ploty**2 + right.best_fit[1]*ploty + right.best_fit[2]

    # recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(img, np.int_([pts]), (0,80,0))
    return img

def decorate_pixels(img):
    img[left.pixels[0], left.pixels[1]] = [255, 0, 0]
    img[right.pixels[0], right.pixels[1]] = [255, 0, 0]
    return img


######################
#        MAIN
######################

cam = Camera(testimg='test_images/test5.jpg')
cam.visualize_calibration()
cam.visualize_distortion()
cam.visualize_perspective()
cam.undistort_test_images()
cam.demo_image()

video = 'project_video.mp4'
white_output = video[0:-4] + '_output.mp4'
clip1 = VideoFileClip(video)#.subclip(21,23)
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

video = 'challenge_video.mp4'
white_output = video[0:-4] + '_output.mp4'
clip1 = VideoFileClip(video)#.subclip(48,55)
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

video = 'harder_challenge_video.mp4'
white_output = video[0:-4] + '_output.mp4'
clip1 = VideoFileClip(video)#.subclip(0,1)
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)
