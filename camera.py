import os
import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from imgproc import *

class Camera():
    def __init__(self, testimg='test_images/test5.jpg', fit_line_in_warped = True):
        # values for perspective transform
        self.x = (163, 540, 768, 1139)
        self.y = (720, 489)
        self.nx = (651-100, 651+100)
        self.ny = (720, 489)

        self.M = None
        self.Minv = None

        self.testimg = testimg

        if os.path.isfile('calib_data.p'):
            # calibration and perspective calculation is already done
            calib_pickle = pickle.load(open('calib_data.p', 'rb'))
            self.mtx = calib_pickle['mtx']
            self.dist = calib_pickle['dist']
            self.M = calib_pickle['m']
            self.Minv = calib_pickle['minv']

        else:
            # calibrate and get perspective matrices, then save data
            self.calibrate()
            self.calc_perspective()

            calib_data = {
                'mtx': self.mtx,
                'dist': self.dist,
                'm': self.M,
                'minv': self.Minv }
            pickle.dump(calib_data, open('calib_data.p', 'wb'))

    def calibrate(self, show_results=False):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        grid = (9,6)
        objp = np.zeros((grid[0]*grid[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:grid[0], 0:grid[1]].T.reshape(-1,2)
        # arrays to store object points and image points from all the images
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane
        # make a list of calibration images
        images = glob.glob('camera_cal/calibration*.jpg')

        # step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, grid,None)
            # if found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

            if show_results:
                # draw and display the corners
                img = cv2.drawChessboardCorners(img, grid, corners, ret)
                cv2.imshow('img',img)
                cv2.waitKey(500)

        if show_results:
            cv2.destroyAllWindows()

        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    def undistort(self, img):
        # return undistorted image based on camera calibration
        return cv2.undistort(img, self.mtx, self.dist)

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

    def visualize_calibration(self):
        img = cv2.imread('camera_cal/calibration1.jpg')
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(img_gray, (9,5), None)
        img = cv2.drawChessboardCorners(img, (9,5), corners, ret)

        img_gray = self.undistort(img_gray)
        img_out = np.dstack((img_gray, img_gray, img_gray))
        ret, corners = cv2.findChessboardCorners(img_gray, (9,5), None)
        img_out = cv2.drawChessboardCorners(img_out, (9,5), corners, ret)

        f, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].set_title('Original', fontsize=16)
        ax[0].imshow(img)
        ax[1].set_title('Undistorted', fontsize=16)
        ax[1].imshow(img_out)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.show()
        f.savefig('output_images/demo_calibration.jpg')

    def visualize_distortion(self):
        img = cv2.imread(self.testimg)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img2 = self.undistort(img)
        f, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].set_title('Original', fontsize=16)
        ax[0].imshow(img, cmap='gray')
        ax[1].set_title('Undistorted', fontsize=16)
        ax[1].imshow(img2, cmap='gray')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.show()
        f.savefig('output_images/demo_distortion_correction.jpg')

    def visualize_perspective(self):
        img = cv2.imread(self.testimg)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = self.undistort(img)
        wimg = self.warp(img)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        wimg = cv2.cvtColor(wimg,cv2.COLOR_GRAY2RGB)
        x, y, nx, ny = self.x, self.y, self.nx, self.ny
        img = cv2.line(img,(x[0],y[0]),(x[1],y[1]),(255,0,0),2)
        img = cv2.line(img,(x[1],y[1]),(x[2],y[1]),(255,0,0),2)
        img = cv2.line(img,(x[2],y[1]),(x[3],y[0]),(255,0,0),2)
        wimg = cv2.line(wimg,(nx[0],ny[0]),(nx[0],ny[1]),(255,0,0),2)
        wimg = cv2.line(wimg,(nx[0],ny[1]),(nx[1],ny[1]),(255,0,0),2)
        wimg = cv2.line(wimg,(nx[1],ny[1]),(nx[1],ny[0]),(255,0,0),2)
        f, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].set_title('Undistorted', fontsize=16)
        ax[0].imshow(img)
        ax[1].set_title('Warped (bird\'s eye)', fontsize=16)
        ax[1].imshow(wimg)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.show()
        f.savefig('output_images/demo_perspective.jpg')

    def undistort_test_images(self):
        images = glob.glob('test_images/*.jpg')
        for fname in images:
            img = cv2.imread(fname)
            corr_img = self.undistort(img)
            cv2.imwrite('output_images/' + fname.split('/')[-1], corr_img)

    def demo_image(self):
        img = cv2.imread(self.testimg)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = self.undistort(img)
        timg = threshold(img, show_results=False)
        #timg = crop_roi(timg)
        wimg = self.warp(timg)
        histogram = np.sum(wimg[int(wimg.shape[0]*0.75):,:], axis=0)

        f, ax = plt.subplots(2, 2, figsize=(16, 8))
        ax = ax.ravel()
        ax[0].set_title('Undistorted', fontsize=16)
        ax[0].imshow(img)
        ax[1].set_title('Sobel threshold', fontsize=16)
        ax[1].imshow(timg, cmap='gray')
        ax[2].set_title('Warped', fontsize=16)
        ax[2].imshow(wimg, cmap='gray')
        ax[3].set_title('Histogram', fontsize=16)
        ax[3].plot(histogram)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.show()
        f.savefig('output_images/demo.jpg')

    def get_test_image(self):
        return self.testimg