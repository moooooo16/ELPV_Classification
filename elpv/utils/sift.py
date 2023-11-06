import cv2 as cv
import os
from tqdm import tqdm
from typing import Callable, Optional, List, Tuple
import numpy as np
from sklearn.cluster import KMeans

class SIFT():
    def __init__(self, img_dir, img_path, n_features=None, contrastThreshold=None,
                 processing_funcs: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None) -> None:
        self.img_path = img_path
        self.n_features = n_features
        self.contrastThreshold = contrastThreshold
        self.sift = cv.SIFT_create(self.n_features, self.contrastThreshold)
        
        if processing_funcs:
            self.images = []
            for image in tqdm(self.img_path, desc='Loading images'):
                
                img = cv.imread(os.path.join(img_dir, image), cv.IMREAD_GRAYSCALE)
                
                for processing_func in processing_funcs:
                    img = processing_func(img)
                    
                self.images.append(img)
        else:
            self.images = [cv.imread(os.path.join(img_dir, img), cv.IMREAD_GRAYSCALE) for img in self.img_path]
    
    
    def calculate_desriptor(self, mask = None) -> Tuple[list, list]:
        descriptors = []
        empty_des = []
        kps = []
        for idx, img in enumerate(tqdm(self.images, desc='Calculating descriptors')):
            kp, des = self.sift.detectAndCompute(img, mask)
            descriptors.append(des)
            kps.append(kp)
            if des is None:
                empty_des.append(idx)
        
        return kps, descriptors, empty_des
    
    def build_features(self, descriptors, ks, state, init = 'k-means++', n_init = 10, max_iter = 300, tol = 1e-4):
        
        hists = {}
        models = {}
        for k in ks:
            print(f'Calculating kmeans for k = {k}')
            hist = np.zeros((len(descriptors), k))
            
            kmeans = KMeans(n_clusters=k, 
                            random_state=state,
                            n_init = n_init,
                            init = init,
                            max_iter=max_iter,
                            tol = tol)
            
            kmeans.fit(np.concatenate(descriptors.copy(), axis=0))
            for idx, des in enumerate(tqdm(descriptors, desc=f'Building histogram for k = {k}')):
                if des.shape[0] == 0:
                    print(idx)
                pred = kmeans.predict(des)
                hist[idx] += np.bincount(pred, minlength=k)

            hists[k]= hist
            models[k] = kmeans
            
        return models, hists
    
    def get_features(self, X, kmean, k):
        
        hist = np.zeros((len(X), k))
        for idx, des in enumerate(tqdm(X, desc='Building histogram for k = {k}')):
            pred = kmean.predict(des)
            hist[idx] += np.bincount(pred, minlength=k)
            
        
        return hist
