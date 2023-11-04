from matplotlib import pyplot as plt
import numpy as np

def plot_img_and_hist(img, prob=None, types=None) -> None:
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[1].hist(img.ravel(), bins=256)
    
    axes[0].set_axis_off()
    # axes[1].set_axis_off()
    
    if prob is not None and types is not None:
        axes[0].set_title(f'Prob: {prob}\nTypes: {types}')
    

def strech_img(img) -> np.ndarray:
    c = np.min(img)
    d = np.max(img)

    return np.clip((img-c) * (255/(d-c)), 0, 255).astype(np.uint8)

def combine_labels(prob, types):
    
    prob_matching = {
        0.0: 0,
        0.3333333333333333: 1,
        0.6666666666666666: 2,
        1.0: 3
    }
    
    types_matching = {
        "poly": 0,
        'mono': 1   
    }
    
    labels = []
    
    for p, t in zip(prob, types):
        labels.append(prob_matching[p] if types_matching[t] else 4 + prob_matching[p])
    
    count, unique = np.unique(labels, return_counts=True)
    print(count, unique)
    return labels