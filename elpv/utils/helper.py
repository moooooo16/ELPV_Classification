from matplotlib import pyplot as plt
import numpy as np
import logging

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


def my_logger(path):
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M')

    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)

    return logger