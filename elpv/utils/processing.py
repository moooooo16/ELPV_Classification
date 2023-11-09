import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern

def strech_img(img) -> np.ndarray:
    c = np.min(img)
    d = np.max(img)

    return np.clip((img-c) * (255/(d-c)), 0, 255).astype(np.uint8)

def clach_img(img, clipLimit, tileGridSize):
    clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img)

def guassian_blur(img, kernel_size, sigmaX):
    return cv.GaussianBlur(img, kernel_size, sigmaX)

def lap_feature(img, dst, ksize):
    return cv.Laplacian(img, dst, ksize)


def morpo_opening(img, kernel, iterations):
    return cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=iterations)

def lbp(img, n_points, radius):
    return np.uint8(local_binary_pattern(img, n_points, radius))


def min_max_normalize(img):
    return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
def flip_y(img):

    return cv.flip(img, 1)


def flip_x(img):

        
    return cv.flip(img, 0)


def rotate(img):

    return cv.rotate(img, cv.ROTATE_180)