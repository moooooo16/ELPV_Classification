from skimage.feature import hog
import cv2 as cv
import os
from tqdm import tqdm

class HOG():
    def __init__(self, img_dir, img_path) -> None:
        self.img_dir = img_dir
        self.img_path = img_path
        
        self.images = [cv.imread(os.path.join(img_dir, img), cv.IMREAD_GRAYSCALE) for img in self.img_path]
    
    def get_hog_features(self, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True):
        
        descriptors = []
        hog_imgs = []
        for idx, img in enumerate(tqdm(self.images, desc='Calculating descriptors')):
            
            fd, hog_img = hog(img,orientations=orient,pixels_per_cell=(pix_per_cell, pix_per_cell),
                cells_per_block=(cell_per_block, cell_per_block),block_norm='L2',
                visualize=vis, feature_vector=feature_vec)
        
            descriptors.append(fd)
            hog_imgs.append(hog_img)
            
        return descriptors, hog_imgs
