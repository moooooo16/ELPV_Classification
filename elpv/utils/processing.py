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
def standardize(img):
    
    return (img - np.mean(img)) / np.std(img)

def flip_y(img):

    return cv.flip(img, 1)


def flip_x(img):

    return cv.flip(img, 0)


def rotate(img):

    return cv.rotate(img, cv.ROTATE_180)

def my_triangle(img):

    count, bins = np.histogram(img.ravel(), bins=255, range=(0, 255))
    # p1: max count, with its index
    p1 = np.array([np.argmax(count), np.max(count)])
    temp = np.max(np.nonzero(count))
    
    # p2: last non-zero count, with its index
    p2 = np.array([temp, count[temp]])
    line = p2-p1
    max_d = -1
    best_p3 = None
    # bins: 0:255, count: count of each pixel
    for p3 in zip(bins[p1[0]:p2[0]+1], count[p1[0]:p2[0]+1]):
        p3 = np.array(p3)
        d = np.linalg.norm(np.cross(line,p1-p3)/np.linalg.norm(line))
        if d > max_d:
            max_d = d
            best_p3 = p3
    threshold = best_p3[0]

    return (img > threshold).astype(np.uint8) * 255

def my_isodata(img):
    t = threshold = np.random.randint(0, 256)
    allowance = 0.0000001

    while True: 
        left = img[img <= threshold]
        right = img[img > threshold]
        
        mu0 = mu1 = 0
        
        if len(left) != 0:
            
            mu0 = np.mean(left)
            
        if len(right) != 0:
            mu1 = np.mean(right)
            
        threshold = (mu0 + mu1) / 2
        if abs(t - threshold) < allowance:
            break
        t = threshold
        
    return (img > threshold).astype(np.uint8) * 255

def my_otsu(img):
    best_threshold = 0
    best_sigma = 0
    for threshold in range(256):
        
        mu0 = mu1 = 0
        p0 = np.sum(img <= threshold) / img.size
        p1 = np.sum(img > threshold) / img.size
        if p0 != 0:
            mu0 = np.mean(img[img <= threshold])
        if p1 != 0:
            mu1 = np.mean(img[img > threshold])
        sigma = p0 * p1 * (mu0 - mu1) ** 2
        
        if sigma > best_sigma:
            best_sigma = sigma
            best_threshold = threshold
    return (img > best_threshold).astype(np.uint8) * 255