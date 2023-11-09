from skimage.feature import hog
import cv2 as cv
import os
from tqdm import tqdm
from typing import Callable, Optional, List, Tuple
import numpy as np
from functools import partial
from concurrent.futures import ThreadPoolExecutor

class FeatureExtraction():
    def __init__(self, img_dir, img_path, label) -> None:
        self.img_dir = img_dir
        self.img_path = img_path
        self.label = np.array(label)
        with ThreadPoolExecutor(max_workers=4) as executor:
 
            self.images = np.array(list(executor.map(self.load_image, [img_dir]*len(img_path), img_path)))
        self.flag = True
        self.sift = cv.SIFT_create()


    def load_image(self, img_dir, img_path):
        return cv.imread(os.path.join(img_dir, img_path), cv.IMREAD_GRAYSCALE)
    
    
    def split_data(self,spliter, randome_state, stratify, split_ratio):
        if stratify:
            X_train, X_test, y_train, y_test = spliter(self.images, self.label, stratify=self.label, test_size=split_ratio, random_state=randome_state)
        else:
            X_train, X_test, y_train, y_test = spliter(self.images, self.label, test_size=split_ratio, random_state=randome_state)
        
        return X_train, X_test, y_train, y_test
        
    def augmentation(self, X_train, labels, augment_funcs):
        augment_x = []
        augment_y = []  

        for img, label in zip(tqdm(X_train, desc="Augmenting images"), labels):
            if label in [0,3,4,7]:
                continue
            
            for augment_func in augment_funcs:
                augmented_img = augment_func(img)
                augment_x.append(augmented_img)  
                augment_y.append (label)   
        
        print(f'Augmenting done, added image: {len(augment_x)}')
        
        self.images = np.concatenate((X_train, augment_x))
        self.label = np.concatenate((labels, augment_y))
        
        return self.images, self.label
    
    def preprocess(self, data, preprocess_funcs):
        out = []
        for image in tqdm(data,  desc='Pre-processing images'):
    
            for processing_func in preprocess_funcs:
                image = processing_func(image)
                if self.flag:
                    func_name = processing_func.func.__name__ if isinstance(processing_func, partial) else processing_func.__name__
                    print(f"After {func_name}: range {image.min()} to {image.max()}")
                    
            self.flag = False
                    
            out.append(image)
            
        self.images = np.array(out)
            
        return self.images
        
    def get_hog_features(self, data, orient, pix_per_cell, cell_per_block, block_norm = 'L2-Hys', vis=False, feature_vec=True):
        
        hog_des = []
        hog_image = []
        for img in tqdm(data, desc='Calculating descriptors'):
            
            hog_out = hog(img,orientations=orient,
                              pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cell_per_block, cell_per_block),
                              block_norm=block_norm,visualize=vis, feature_vector=feature_vec)
        
            if vis:
                hog_out, hog_img = hog_out
                hog_image.append(hog_img)
                hog_des.append(hog_out)
            else:
                hog_des.append(hog_out)
            
        if vis:
            return np.array(hog_des), np.array(hog_image)
        else:
            return np.array(hog_des)
        
    def get_sift_descriptor(self,data, mask = None) -> Tuple[list, list, list]:
        descriptors = []
        empty_des = []
        kps = []
        for idx, img in enumerate(tqdm(data, desc='Calculating descriptors')):
            kp, des = self.sift.detectAndCompute(img, mask)
            descriptors.append(des)
            kps.append(kp)
            if des is None:
                empty_des.append(idx)
        
        return kps, descriptors, empty_des

    
    
    def build_sift_cluster(self,kmean_clf, descriptors, ks, state, init = 'k-means++', n_init = 10, max_iter = 300, tol = 1e-4):
        hists = {}
        models = {}
        for k in ks:
            print(f'Calculating kmeans for k = {k}')
            hist = np.zeros((len(descriptors), k))
            
            
            
            kmeans = kmean_clf(n_clusters=k, 
                            random_state=state,
                            n_init = n_init,
                            init = init,
                            max_iter=max_iter,
                            tol = tol)
            
            kmeans.fit(np.concatenate(descriptors.copy(), axis=0))
            
            for idx, des in enumerate(tqdm(descriptors, desc=f'Building histogram for k = {k}')):
                if des.shape[0] == 0:
                    print(f"Empty descriptor at index {idx}")
                    continue
                pred = kmeans.predict(des)
                hist[idx] += np.bincount(pred, minlength=k)

            hists[k]= hist
            models[k] = kmeans
            
        return models, hists
    
    def get_hist(self, descriptor, kmean, k):
        
        hist = np.zeros((len(descriptor), k))
        for idx, des in enumerate(tqdm(descriptor, desc=f'Building histogram for k = {k}')):
            if des.shape[0] == 0:
                print(f"Empty descriptor at index {idx}")
                continue
            pred = kmean.predict(des)
            hist[idx] += np.bincount(pred, minlength=k)
            
        
        return hist