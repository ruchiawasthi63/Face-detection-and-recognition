# Face-detection-and-recognition
This code is in response of a 3 days assignment that was given to me that consisted of two stages:
### Problem Statement: Build a face match authentication system to validate the ID proof.
###### Task 1: Face Detection
-Build a face detector to identify faces in both the images.
I/P: Image
O/P: Face boundary (bounding box) and confidence score (i.e. how sure the algorithm is that it is a
face)
###### Task 2: Face Recognition
-Given Image_1, check if the same person is present in Image_2 and if yes provide a metric as to how
good the match is ( i.e. a confidence score)
I/P: (Image_1 , Image_2)
O/P: Match (Yes or No) , Confidence Score

## Code 
Functions that have been used are:
###### 1. convertToRgb -
INPUT: BGR image, OUTPUT: RGB image (converts BGR image into RGB image)

###### 2. confidence_Score - 
INPUT: Number of faces detected corresponding to min_neigbors = 5, min_neigbors corresponding to number of faces=1, OUTPUT: Confidence score for face detection (gives confidence score of face detection)

###### 3. face_detection - 
OUTPUT: image with bounding box around the predicted face and confidence score

###### 4. ROI - 
INPUT: The address of the image that users enter (img), The name by which the cropped image will get saved (name), OUTPUT: returns cropped image which consists the ROI (Region of Interest) i.e. the area where face has been detected

###### 5. task2 - 
OUTPUT: returns whether or not the two images passed are of similar person - True for Yes and False for No, confidence score of recognition 


### Why haar classifier?
There are primarily two pre trained face detectors that Open CV provides:
1. HAAR Classifier
2. LBP Classifier

But it has been found that HAAR Classifier gives better accuracy over images with a little trade off with the speed as compared to when using LBP classifer.
https://www.superdatascience.com/blogs/opencv-face-detection

### Justification of confidence score for face detection?
- There isn't any straightforward way present to calculate the confidence score as such for HAAR classifier. So I had to intutively think of some confidence score formula that could best showcase the trend of my predictions.
- After experimentally passing a lot of images through my face detection function I found that if the min_neigbors are too less there are chances of overfitting and the detector detects very small faces. If the min_neigbors are too much then the detector tends to miss out on important information, here, face. Generally min_neigbors = 5 gives the best results.
- When I passed nearly 50 images through the function it was found that if the detetcor detects 1 face corresponding to min_neigbors = 5, in all those 50 cases it had detected the correct face hence confidence score = 100%
- My general formula for calculating confidence score was 
-         (y2-y1)/(x2-x1)
where,
- y2 = Number of faces corresponding to when min_neigbors = 5
- y1 = number of faces = 1
- x2 = Number of min_neigbors when finally it detects 1 face
- x1 = min_neigbors = 5 because that is considered as our datum
.This is then multiplies by 3 since we're moving with a stride of 3 i.e. we're checking for every third min_neigbor
