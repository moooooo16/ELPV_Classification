# Usage

Go to root directory of the project and
Use `pip install -e .` to install the package in editable mode.


# Ideas
- Simple binary thresholding
- SIFT and BOVW
- A NN with:
  - MobileNet V2 structre
  - EffecientNet w, d, resolution
  - Designed for 
    - gray scale images
    - Small images
    - small models
  


# Findings
- Data is heavily unbalanced

# Results

K : 64 -> 1024, dense -> sparse

K = 64
KNN: 0.67, 0.63
SVM: 0.68, 0.64
RF: 0.68, 0.65
LogReg: 0.65, 0.62
Tree: 0.61, 0.63
adaB: 0.59, 0.60
GdtB: 0.68, 0.62

K = 128
KNN: 0.66, 
SVM: 0.69
RF: 0.66

K = 256
KNN: 0.65
SVM: 0.68
RF: 0.67

K = 384
KNN: 0.61
SVM: 0.70
RF: 0.64

K = 512
KNN: 0.57
SVM: 0.71
RF: 0.65

K = 640
KNN: 0.52
SVM: 0.66
RF: 0.67

K = 768
KNN: 0.44
SVM: 0.68
RF: 0.68

K = 896
KNN: 0.42
SVM: 0.71
RF: 0.68


# Features

0. Mono + 0.0
   - Little verticle line
   - 3 horizontal lines
     - Dashed or solid
   - 4 Corners are black
   - No bolbs or cluster of shade
1. Mono + 0.33
   - Small scratch/shade
   - 3 Horizontal lines
   - 
2. Mono + 0.66
3. Mono + 1.0
4. Poly + 0.0
5. Poly + 0.33
6. Poly + 0.66
7. Poly + 1.0


# Method 1

- Use ICA to find a set of representative basis images that can best describe various defect-free solar cell sub-images
- Each other image is presented as a linear combination of these basis images
- In detection, feature extraction and image construction from the basis images are used to evaluate the preesence/absence of defects in the inspection image.

- Preprocessing:
  - Morphological smoothing
    - Design three structuring elements in the directions of 45, 90, and 135 degrees.
    - Each length L pixels and width 1 pixel
    - Random dark region dimension < L < defect length 
    - Dark regions can be smoothed, defects can be preseved.
    - L is givin by **13** in the paper.
  - For a given pixel point with gray level , the gray levels in each of the three SEs are accumulated.
  - ![image](/project/elpv/doc/L_function.png)
  - ![L_function2](/project/elpv/doc/L_function2.png)
  - Select $\theta-SE = argmin(S_{45}, S_{90}, S_{135})$
    - This representing minimum accumulated magnitude amoung 3 dirictions 
  - Apply gray-level diliation (local maximum gray level) is applie to the image with selected SE in direction $\theta$
  - If $\theta-SE$ contains all defect points, the dilated value will be still small for dark region, other wise  
  - Horizontal bus bar should be removed.
    - OP methioned didnt include 0 degree SE, but I think it should be included.
- ICA model
  - Mixture signals X can be represent as $X = A \cdot S$
  - A is unknown mixing matrix, S is unknown source matrix
  - ICA is obtained by finding demixing matrix W such that $U = W \cdot X$
    - where U is the independent components(IC)
    - fastICA can be used.
- Find "Basic images"
  - Randomly chosen from the training set(free from defect)
  - Reshape each image to 1 D row vector in $X$
  - X contains B samples, X is shape of (B,  N)
- Represent test image with b
  - 1D test image is y
  - $y  = b \cdot U = \sum_{i=1}^{B}b_i \cdot u_i$
  - b is coefficient vector of the linear combination, U is basic images
  - b = (b1, b2, ..., bB)
  - b is obtained by solveing $b = y \cdot U^+$
  - $U^+ = U^T(U \cdot U^T)^{-1}$
- Deterin the defects
  - Use cosine distance to evaluate similarity between 
  - ![cosine](/project/elpv/doc/cos_simi.png)
  - $b_i  = x_i \cdot U^+$
  - if cosine distance is zero -> identical
  - final distance is smallest distance among all basic images
  - If distance is > than threshold, it is defect.
  - 