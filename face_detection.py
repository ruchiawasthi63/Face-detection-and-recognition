
# coding: utf-8

# In[ ]:


def face_detection(): 
    image_location = input("Enter the location of your image")
    #image_location has to be of the form - 'F:/selfie/Selfie/adhaark.jpg'
    #read the image
    test = cv2.imread(image_location)
    
    #resize the image in size 512x512
    test= cv2.resize(test, (512,512))
    
    #convert the image in grayscale
    gray_img = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    
    #Iterating over different values of min_neigbors to find the best of them 
    for neighbors in range(5,50,3):
        faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=neighbors)
        faces_5 = len(haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5))
        
        
        if len(faces)==1:
            print ('1 face found corresponding to' , neighbors, 'neighbors')
            for (x, y, w, h) in faces:
                cv2.rectangle(test, (x, y), (x+w, y+h), (0, 255, 0), 2)
                plt.imshow(convertToRGB(test))
                plt.show()
                
                
                if neighbors==5:
                    print ('ConfidenceScore = 100%')
                    
                else:
                    print ('ConfidenceScore =' ,confidence_Score(faces_5,neighbors) , '%')
            break
            
    if len(faces)>1:
        #it is able to detect but there are more than one faces in the image
        print ("Face Undetectable. Upload another image")
        
    if len(faces)==0:
        print ("No face found")


# ###### Why iterating over different min_neighbors?
# This was done due to the fact that not all images can detect face when min_neighbor is set to 5. Also, to detect that one most prominent face one will have to check for different values of min_neighbor. Some might detect more than one faces in a smaller value of min_neighbor but might give accurate face if min_neigbours is increased. 
# 
# 
# when min_neighbor runs in a for loop and one prominent face is detected at a larger min_neighbor value
# 
# 
# ###### - len(faces) = Number of faces detected in the image
# ###### - len(faces) >1 : 
# This might occur because the image may contain group selfies or patterns that look like faces. As we increase the number of min_neigbors the chances of predicting false positives decreases and it is able to detect the face that is most prominent in the image. Despite 50 min_neigbors if it could not detect the number of faces as 1 then we say that the face is not detectable in the image
# ###### - len(faces) =0: 
# This might occur when there is either no image or image is in certain orientation that it is not able to detect it. The same image when rotated was able to detect a face. 
# 
