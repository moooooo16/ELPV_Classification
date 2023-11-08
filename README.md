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