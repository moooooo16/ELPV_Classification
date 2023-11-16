from skimage.feature import hog
import cv2 as cv
import os
from tqdm import tqdm
from typing import Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from sklearn.model_selection import KFold

class FeatureExtraction():
    def __init__(self, img_dir, img_path, label, l = 15) -> None:
        self.img_dir = img_dir
        self.img_path = img_path
        self.label = np.array(label)
        with ThreadPoolExecutor(max_workers=4) as executor:
 
            self.images = np.array(list(executor.map(self.load_image, [img_dir]*len(img_path), img_path)))
        self.flag = True
        self.sift = None
        self.ses = self.create_SE(l)

    def load_image(self, img_dir, img_path):
        return cv.imread(os.path.join(img_dir, img_path), cv.IMREAD_GRAYSCALE)
    
    
    def split_data(self,spliter, randome_state, stratify, split_ratio):
        if stratify:
            X_train, X_test, y_train, y_test = spliter(self.images, self.label, stratify=self.label, test_size=split_ratio, random_state=randome_state)
        else:
            X_train, X_test, y_train, y_test = spliter(self.images, self.label, test_size=split_ratio, random_state=randome_state)
        
        return X_train, X_test, y_train, y_test
        
    def augmentation(self, X_train, labels, augment_funcs):
        if len(augment_funcs) == 0:
            return X_train, labels
        
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
        
        out_images = np.concatenate((X_train, augment_x))
        out_labels = np.concatenate((labels, augment_y))
        
        return out_images, out_labels
    
    def preprocess_single_image(self, args):
        image, preprocess_pipeline = args
        for preprocess_step in preprocess_pipeline:
            processing_func, func_params = preprocess_step
            image = processing_func(image, **func_params)
        return image
    
    def preprocess(self, data, preprocess_pipeline):
        if len(preprocess_pipeline) == 0:
            return data
    
        args = [(image, preprocess_pipeline) for image in data]

  
        with Pool() as pool:
   
            results = list(tqdm(pool.imap(self.preprocess_single_image, args), total=len(args), desc='Pre-processing images'))


        return np.array(results)

    def old_preprocess(self, data, preprocess_pipeline):
        out = []
        for image in tqdm(data,  desc='Pre-processing images'):
    
            for preprocess_step in preprocess_pipeline:
                processing_func, func_params = preprocess_step
                image = processing_func(image, **func_params)
                    
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
        if self.sift == None:
            self.sift = cv.SIFT_create()
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
    
    
    def kaze_detetion(self):
        
        pass
    
    
    def vgg_descriptor(self):
        pass    
    
    def create_SE(self, l=15):
        
        se45 = np.zeros((l, l), dtype=np.uint8)
        for i in range(l):
            se45[i, l-i-1] = 1
        
        se135 = np.rot90(se45.copy(), 1)
        
        se90 = np.zeros((l, l), dtype=np.uint8)
        for i in range(l):
            se90[i, l//2] = 1
        
        return se45, se90, se135
    def calculate_threshold(self, kf, X_train, y_train, indices):
        
        non_defecet_y = y_train[indices]
        non_defecet_X = X_train[indices]
        
        mean_error = []
        std_error = []

        for i, (train_index, test_index) in enumerate(kf.split(non_defecet_X, non_defecet_y)):
            train_x = non_defecet_X[train_index]
            test_x = non_defecet_X[test_index]

            X = train_x.reshape(train_x.shape[0], -1)
            X_test = test_x.reshape(test_x.shape[0], -1)

            errors = self.calculate_error(X, X_test)

            mean_error.append(np.mean(errors))
            std_error.append(np.std(errors))
            
            print(f'Fold {i+1} mean error: {np.mean(errors)}, std error: {np.std(errors)}')
        
        return np.mean(mean_error), np.mean(std_error)
    
    def calculate_error(self, X, y):
        
        X = X.reshape(X.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        

        X_plus = np.linalg.pinv(X)

        y_hat = np.dot(np.dot(y, X_plus), X)
  
        norm_y = np.linalg.norm(y, axis=1)
        nomr_y_hat = np.linalg.norm(y_hat, axis=1)
        nomr_y_hat[nomr_y_hat == 0] = np.finfo(float).eps
        
        C = norm_y / nomr_y_hat
 
        C = C.reshape(-1, 1)


        errors = np.linalg.norm(y - C * y_hat, axis=1)
        

        return errors
        