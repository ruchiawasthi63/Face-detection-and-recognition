
# coding: utf-8

# ## Code flow (ROI)
# This function is to be used when face recognition is also to be performed. It works similar to the face_detetcion function but additionally stores the cropped image ( image surrounded by the bounding box) for further comparision.

# In[ ]:


def ROI(img,name): 

    test = cv2.imread(img)
    test= cv2.resize(test, (512,512))
    gray_img = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    for neighbor in range(5,50,3):
        faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=neighbor)        
        
        if len(faces)==1:

            for (x, y, w, h) in faces:
                cv2.rectangle(test, (x, y), (x+w, y+h), (0, 255, 0), 2)

                test=test[y:y+h,x:x+w]
                plt.imshow(convertToRGB(test))
                plt.show()

                test= cv2.resize(test, (222,222))
                im = Image.fromarray(test)
                img = img.replace((img.split('/')[-1]) , "")
                im.save(img+name+'.jpg')
                
                
            break
            
    if len(faces)>1:
        #it is able to detect but there are more than one faces in the image
        print ("Face Undetectable. Upload another image")
        
    if len(faces)==0:
        #it is able to detect but there are more than one faces in the image
        print ("No face found")

