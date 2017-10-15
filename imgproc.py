import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_roi(img):
    # only keeps the region of the image defined by the polygon
    (h, w) = img.shape
    rect = np.array( [[[0,h],[0,h*0.75],[w*0.4,h*0.55],[w*0.6,h*0.55],[w,h*0.75],[w,h]]], dtype=np.int32 )
    # defining a blank mask to start with
    mask = np.zeros_like(img)
    # filling pixels inside the polygon defined by "rect" with the fill color
    cv2.fillPoly(mask, rect, 255)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def sobel_thresh(img, orient='x', thresh=(0, 255), sobel_kernel=3):
    # calculate x or y Sobel gradient, rescale to 8 bit and threshold
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary

def magnitude_thresh(img, thresh=(0, 255), sobel_kernel=3):
    # calculate Sobel gradient magnitude, rescale to 8 bit and threshold
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary = np.zeros_like(gradmag)
    binary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return binary

def direction_thresh(img, thresh=(0, np.pi/2), sobel_kernel=3):
    # calculate Sobel gradients and theur direction, then threshold
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary =  np.zeros_like(absgraddir)
    binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary

def plot_results(img, gray1, gray2=None, str1=None, str2=None, str3=None, file=None):
    cnt = 3 if gray2 is not None else 2
    f, ax = plt.subplots(1, cnt, figsize=(16, 5))
    if str1 is not None: ax[0].set_title(str1, fontsize=16)
    ax[0].imshow(img)
    if str2 is not None: ax[1].set_title(str2, fontsize=16)
    ax[1].imshow(gray1, cmap='gray')
    if gray2 is not None:
        if str3 is not None: ax[2].set_title(str3, fontsize=16)
        ax[2].imshow(gray2, cmap='gray')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()
    if file is not None: f.savefig('output_images/' + file)

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

    if show_results:
        plot_results(img, combined,
            str1='Original',
            str2='Sobel gradients X and Y on grayscale and S-channel',
            file='sobel_threshold.jpg')


    return combined

