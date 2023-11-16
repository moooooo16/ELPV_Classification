# ELPV dataset classification

## Introduction

### Challenges

This dataset contains 2 types of image, namely mono and poly. Each type further divided to 4 classes, each class representing different degree of defect. The dataset is highly imbalanced, with the number of "0,0" defect is dominating the databse (about half of the dataset). The dataset is also quite noisy, its really hard for human to distinguish the image especially for "0.33" and "0.67", and they are look very similar. 


In literature review section, we will briefly discuss the related work in this field, and method I choose or considered to complete the task. In method section, I will provide in-depth discussion.

## Literature Review

Defect detection for EL images is active topic in recent years. There are several different methods proposed to solve this problem. Traditional machine learning method is used to solve this problem. Generally it can be described as following steps: pre-processing image for better feature extraction, feature extraction and selection, and then use a classifier to classify defect and non-defect. <cite>[Sergiu][1]</cite> follows the tradition method, tested with different feature extraction method and use SVM and CNN as classifier. Another very interesting approach proposed by <cite>[Tsai][2]</cite> uses ICA to extract the basis image, and then calculate similarity between testing image and basis image. This method can only detect defect/non-defect, but can not find percentage defect. I also included this method in the code, so I can give a taste of different approaches. 

When dealing with datasets, class-imbalanced problem is significant. SVM on the other hand is very sensitive to noises. Chapter 5 of <cite>[He][3]</cite> gives a very good overview of how to deal with class-imbalanced problem. Imbalance problem in SVM causing decision boundray biased toward the majority class, eg. more testing sample will be classified as majority class. To solve this problem, this book provided several methods such as, uses different sampling method(upsampling, and downsampling), apply class ratio to parameter C (Different error cost). Rehan Akbani <cite>[Rehan][4]</cite> also proposed several methods to solve this problem, such as use SMOTE method for upsampling, and use different error cost to enhance the performance. 

Features extraction is also very important for this problem. In this project, I examined several different features extraction method. SIFT, SURF, ORB and KAZE are most common descriptors used in computer vision. Shaharyar <cite>[Shaharyar][5]</cite> compared all methods mentioned above in different aspects, and result shows that SIFT are found to be most accurate algorithm on average. <cite>[Sergiu][1]</cite> further compared SIFT and SURF, and the result shows SIFT perform well on this dataset.


## Traditional Methods

The first method I use is following Tsai's reconstruction method <cite>[Tsai][2]</cite>. From some basis data analysis, I found that EL image have randomly-shaped dark regions in the background, and defect are dark line shape or bar shape regions, so first thing I want to do is to preprocess the image so these randomly shaped region can be removed. Meanwhile, we also need to retain the line-shape or bar-shape defects. Tsai proposed to use morphological smoothing. 3 structuring elements in direction of 45, 90, and 135 are used for morphological smoothing. Each SE are length of 15 (13 in original paper). This length is longer than most random shaped regions but shorter than the cracks. For each pixel in the image, we select the SE with minimum accumulated gray levels(sum of intensity within each SEs). The smallest direction will be selected as morphological dilation smoothing. Quote from original paper: "If -SE contains all defect points, the dilated value will be still small for a dark region. Otherwise, the dilated value will be large for a bright region in the background."[2] This dilation is basicly select the highest intensity value and replace the original pixel value. This preprocessing method is very efficiency to remove the background noise and retain the defect. 

Tsai believe that the test image can be reconstructed by linear combination of basis images. The basis images are set of non-defect images from the dataset("0%" being defect). If the test sample contains defects, then it is expected that the error should be large. The determination of defect is then based on a pre-defined threshold. If error is greater than this threshold, then the test image are considered as defect. My original idea is to set 3 different threshold to classify different type of defect, but this later prove to be not a effective method. To derive this basis images, we can use FastICA. ICA can separate noise from observed signal, $U = W \cdot X$ where $X$ is observed signal, W is the mixing matrix, and U is the source signal. In our case, U is the basis images. All test images can be reconstruct by $\hat{y} = b \cdot U$ where b is the coefficient vector of linear combination. $b$ can be obtained from $b = y \cdot U^+$, $U^+$ is pseduo-inverse of $U$. Since we can reconstruct the test image, we can rewrite this function to following equation: 
$$
\begin{align}
\hat{y}& =b \cdot U \\& = (y \cdot U^+) (WX) \\& = y(WX)^T[(WX) \cdot (WX)^T]^{-1}(WX) \\& = y \cdot X^T[X \cdot X^T]^{-1}X \\& = (y \cdot X^+)\cdot X\\
\end{align}
$$
With above equation, we don't have to calculate ICA for each test image, and we can just use original defect-free dataset to reconstruct the test image. Threshold can be determined by using same method. I spited non-defect dataset into 5 folds, and use 4 folds as $X$, and remaining one fold to calculate the mean error. Error are caluclated as following: $\Delta \epsilon(y) = ||y - c \cdot \hat{y}||, c=\frac{||y||}{||\hat{y}||}$ c is used asa regularization term. Now we can calculate the mean error for each test images, and use this error to classify images.

The second method I use is standard feature extraction and classification method. 

## Experimental Results

## Discussion

## Conclusion

## References
[1] Deitsch, S., Christlein, V., Berger, S., Buerhop-Lutz, C., Maier, A., Gallwitz, F., &amp; Riess, C. (2019). Automatic classification of Defective Photovoltaic module cells in electroluminescence images. Solar Energy, 185, 455–468. https://doi.org/10.1016/j.solener.2019.02.067 

[2] Tsai, D., Wu, S., & Chiu, W. (2013). Defect detection in solar modules using ICA basis images. IEEE Transactions on Industrial Informatics, 9(1), 122–131. https://doi.org/10.1109/tii.2012.2209663

[3] He, H., & Ma, Y. (2013). Imbalanced learning. In Wiley eBooks. https://doi.org/10.1002/9781118646106

[4] Akbani, R., Kwek, S., & Japkowicz, N. (2004). Applying support vector machines to imbalanced datasets. In Lecture Notes in Computer Science (pp. 39–50). https://doi.org/10.1007/978-3-540-30115-8_7

[5] Tareen, S. a. K., & Saleem, Z. (2018). A comparative analysis of SIFT, SURF, KAZE, AKAZE, ORB, and BRISK. 2018 International Conference on Computing, Mathematics and Engineering Technologies (iCoMET). https://doi.org/10.1109/icomet.2018.8346440