# Usage

Go to root directory of the project and
Use `pip install -e .` to install the package in editable mode.


# File Structure


## ELPV
>All files under elpv folder are wrote by Mo Li.

> his folder foucse on the implementation of machine learning algorithm and related functions.


> You will need to use command in USAGE to run the code.

### elpv/archive
Archive are files contains old version of code, including different methods mentioned in the report.

### elpv/data
This folder should include images folder(folder name should be images) and "labels.csv" file. Images folder should contain all images in the dataset.

### elpv/src
This folder contains all main logic of the project.

- EDA/EDA2 are playground to show the effect of all preproecessing methods.

- SVM contains HOG descriptor -> SVM classifier implementation.

- reconstruct contains ICA-like method implementation. First method introduced in the report.

### elpv/utils
This folder contains all utility functions.

- **classifier.py** contains all classifier used in the project.
  - `get_report` -> print classification report and conf mat
  - `down_sampling` and `vote` -> down sampling methods and voting after down sampling
  - `up_sampling` with `smote`, `one_vs_other_up_sampling` and `distance_vote`
    - up sampling use SMOTE to generate new data
    - up sampling use ovo and one vs other method to generate new data
  - `print_misclassifications` and `reconstruct_pred`
    - These functions for reconstrurction method

- **dataloader.py** contains all function needed to load data, and prepare all path for the project.

- **features.py** contains all feature related functions
  - `split_data`, `augmentation`, and `preprocess` 
    - These are piping functins, they will take functions from other files and apply them to the data.
  - `get_hog_features`,  `get_sift_descriptor`, `build_sift_cluster`, `get_hist`
    - These are descriptor related functions, use these function to build feature vecotrs
  - `create_SE`, `calculate_threshold` and `calculate_error`
    - These are reconstruction related functions, use these function for reconstruction method
- **preprocessing.py** file inclue all preprocessing methods.
  - From contract increasing to rotation, thresholding, smoothing, and etc.

### elpv/out
- This folder contains all checkpoints, results, logs, plots, and etc.