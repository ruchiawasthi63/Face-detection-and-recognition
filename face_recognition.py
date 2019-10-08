
# coding: utf-8

# ## Code flow (task2)
# This function does the major part of comparing two images based on the feature vectors produced wrt known image. Once these encodings are compared the algorithm is able to tell the extent to which these images are similar based on the values of encodings. Confidence score is calculated based on the differences in the values of both the encodings. Here as well, the confidence score has been calculated intutively as there is not hard and fast method that exists for calculating the same.

# In[ ]:


def task2():
    selfie_location = input("Enter the location of your selfie")
    #selfie_location has to be of the form - 'F:/selfie/Selfie/image.jpg'
    ROI(selfie_location,'selfie')
    
    aadhar_location = input("Enter the location of your aadhar")
    #aadhar_location has to be of the form - 'F:/selfie/Selfie/image.jpg'
    ROI(aadhar_location,'aadhar')
    
    selfie_location = selfie_location.replace((selfie_location.split('/')[-1]) , "selfie.jpg")
    aadhar_location = aadhar_location.replace((aadhar_location.split('/')[-1]) , "aadhar.jpg")
    
    known_image = face_recognition.load_image_file(selfie_location)
    unknown_image = face_recognition.load_image_file(aadhar_location)

    known_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
    
    difference= known_encoding-unknown_encoding
    sum = 0
    
    for i in range(0,128):   
        sum = sum + (difference[i]*(difference[i]))

    confidence = "%.2f" % round( (100 - ((sqrt(sum/128))*100 )),2)   
    confidence= str(confidence)
    return results, ('Confidence Score = '+ confidence+'% of the prediction being correct')
    

