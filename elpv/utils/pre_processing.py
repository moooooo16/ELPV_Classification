#%%
from tqdm import tqdm
import numpy as np
import pickle
    

# %%
def bovw(clusters, descriptors, method='SK'):
    hist = np.zeros((len(descriptors), clusters.n_clusters))
    
    if method == 'SK':
        for hst, des in tqdm(zip(hist, descriptors), 
                             desc='Building histogram'):
            # des shape: (n_des, 128)
            pred = clusters.predict(des)
            # pred shape: (n_des, )
            # hst shape: (n_clusters, )
            hst += np.bincount(pred, minlength=clusters.n_clusters)
    
    if method == 'L2':
        pass

    return hist


# %%
