from matplotlib import pyplot as plt
import logging


def plot_img_and_hist(img,prob=None, types=None) -> None:

    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap='gray')
    axes[1].hist(img.ravel(), bins=256)
    axes[0].set_axis_off()
    # axes[1].set_axis_off()

    if prob is not None and types is not None:
        axes[0].set_title(f'Prob: {prob}\nTypes: {types}')


def my_logger(path, name):
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M')

    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)

    return logger


def compar_img(orimg, newimg, labels):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(orimg, cmap='gray')
    axes[1].imshow(newimg, cmap='gray')
    if labels is not None:
        axes[0].set_title(f'Prob: {labels[0]}\nTypes: {labels[1]}')
    plt.show()
    
def plot_reconstruct(errors, thresh_mean, thresh_std):
    int_errors = errors.astype(int)
    plt.hist(errors, bins=100, range = (min(int_errors), max(int_errors)))
    plt.vlines(thresh_mean, 0, 40, colors='r', linestyles='dashed')
    plt.vlines(thresh_mean + thresh_std, 0, 40, colors='r', linestyles='dashed')
    plt.vlines(thresh_mean + 2* thresh_std, 0, 40, colors='r', linestyles='dashed')
    plt.vlines(thresh_mean + 3* thresh_std, 0, 40, colors='r', linestyles='dashed')
    plt.show()