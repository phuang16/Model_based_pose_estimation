# Model-based-pose-estimation

**Problem statement:**
  - Given: (1) image of an object (2) location of features on the object (model geometry), and (3) the corresponding location of the features in the image 
  - Find: the pose (position & orientation) of the object with respect to the camera
  - In addition, uncertainy of the estimated pose was obtained by the covariance matrix  

**Method:**
Least square method

**Reference:**
This python code was adapted from the example Matlab codes in Prof. William A. Hoff's lecture (EENG 512/CSCI 512 Computer Vision, https://inside.mines.edu/~whoff/ ), where the sample images were also provided. Permission for adaption was kindly granted. 
