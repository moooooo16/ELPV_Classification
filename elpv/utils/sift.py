import cv2 as cv
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import Union, List
from sklearn.cluster import KMeans
from utils.elpv_reader import DATA_PATH

class Sift():
    def __init__(self,
                images: Union[List[str], List[np.ndarray]] = None, 
                nfeatures=None, 
                contrastThreshold=None, 
                flag = None) -> None:
        
        self.flag = flag
        self.nfeatures = nfeatures
        self.images = []
        
        if isinstance(images[0], str):
            self.images = [cv.imread(DATA_PATH + '/' +img, cv.IMREAD_GRAYSCALE) for img in images]
        
        if isinstance(images[0], np.ndarray):
            self.images = images
            
        self.sift = cv.SIFT_create(nfeatures=self.nfeatures,
                                   contrastThreshold=contrastThreshold)
        
        self.descriptor, self.zero_des, self.des_hist= self.calculate_descriptor(self.images)
    
    def plot_hist(self):
        plt.hist(self.des_hist, bins=self.nfeatures)
        plt.show()
        
        idx, count = np.unique(self.des_hist, return_counts=True)
        print(idx, count)
        
        
    def detect(self, img, mask=None):
        kp, des = self.sift.detectAndCompute(img, mask)
        
        return kp, des
    
    def draw(self, img, kp):
        
        img = cv.drawKeypoints(img, kp, None, flags = self.flag)
        
        return img
    
    def calculate_descriptor(self, images):
        
        descriptors = []
        zero_des = []
        des_hist = []
        
        for idx, img in enumerate(tqdm(images, desc='Calculating descriptors')):
            _, des = self.detect(img)
            
            if des is None:

                des = np.zeros((1,128))
                zero_des.append(idx)
                des_hist.append(0)
                descriptors.append(des)
                continue
            
            des_hist.append(des.shape[0])
            descriptors.append(des)
        
        return descriptors, zero_des, des_hist
    
    def calculate_clusters(self, descriptors, ks, n_init='auto', tol=1e-4):
        
        if len(descriptors) == 0:
            raise ValueError('No descriptors')
        
        if len(descriptors) == self.descriptor:
            raise ValueError('All data for cluster')
        
        
        results = {}
                
        if isinstance(ks, list):
            pass
        
        if isinstance(ks, int):
            ks = [ks]
        
        for k in tqdm(ks, desc='Calculating clusters'):
            kmeans = KMeans(n_clusters=k, 
                            random_state= 0,
                            n_init=n_init,
                            tol = tol)
            
            kmeans.fit(np.concatenate(descriptors.copy(), axis=0), 
                    #callback=[self.displaybar])
            )

            results[k] = kmeans
        
        return results
