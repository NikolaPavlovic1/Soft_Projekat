from __future__ import print_function

import cv2

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump, load

import numpy as np

nbins = 8 
cell_size = (6, 6) 
block_size = (2, 2) 


from keras.datasets import mnist
(trainX, trainy), (testX, testy) = mnist.load_data()

def dilate(image):
    kernel = np.ones((3,3))
    return cv2.dilate(image, kernel, iterations=3)

def trainSVM(pos_features):
    x_train = reshape_data(pos_features)
    y = np.array(trainy[0:40000])
    print(x_train.shape,y.shape)
    
    clf_svm = SVC(kernel='linear', probability=True) 
    clf_svm.fit(x_train, y)
    
    dump(clf_svm, 'digitDetection.joblib')  
    return

def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def load_image_to_hsv(path):
    return cv2.cvtColor(path,cv2.COLOR_BGR2HSV)

def countX(lst, x): 
    count = 0
    for ele in lst: 
        if (ele == x): 
            count = count + 1
    return count

def testAcc():
    m = load('digitDetection.joblib')    
   
    pos_features=[]
    for x in  testX :
            img=x
            hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                          img.shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)
            pos_features.append(hog.compute(img))
    pos_features = np.array(pos_features)
    x_test = reshape_data(pos_features)
    y = np.array(testy)
    y_train_pred = m.predict(x_test)    
    print("Tacnost modela: ", accuracy_score(y, y_train_pred))

testAcc()

m = load('digitDetection.joblib')  

cap = cv2.VideoCapture('video1.mp4')

# video 1 = 1
# video 2 = 5
# video 3 = 7

a=[]
pred = []
zbir=0

while(True):   
        
        ret, frame = cap.read()
        img = image_gray(frame)
        img = 255 - img
        img = dilate(img) 
        imbin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
        img1, contours, hierarchy = cv2.findContours(imbin,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        maximum=0
        for contour in contours: 
            x,y,w,h = cv2.boundingRect(contour)  
            if maximum<w*h and w<300:
                maximum =w*h
        for contour in contours: 
            x,y,w,h = cv2.boundingRect(contour)            
            if maximum==w*h : 
                img = image_gray(frame)
                img = 255 - img  
                img = img[y:y+h,x:x+w]
                hsv = load_image_to_hsv(frame);
                sensitivity = 25;
                lower = np.array([30 - sensitivity, 100+sensitivity, 50])  
                upper = np.array([30 + sensitivity, 255+sensitivity, 255])
                mask = cv2.inRange(hsv,lower,upper);
                imbin = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
                img1, contours, hierarchy = cv2.findContours(imbin,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                neg=False
                if len(contours)>20 :
                    neg=True
                img = cv2.resize(img,(28,28))
                hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                              img.shape[0] // cell_size[0] * cell_size[0]),
                                    _blockSize=(block_size[1] * cell_size[1],
                                                block_size[0] * cell_size[0]),
                                    _blockStride=(cell_size[1], cell_size[0]),
                                    _cellSize=(cell_size[1], cell_size[0]),
                                    _nbins=nbins)
                pos_features=hog.compute(img)
                pos_features = np.array(pos_features).reshape(1,-1)
                
                res=(m.predict(pos_features))
                pred.append(res)
                
                if len(pred)>15:
                    pred.pop(0)
                if countX(pred,res)==15 and res not in a:                        
                    a.append(res)
                    if neg==True:
                        zbir-=res[0]
                    else:
                        zbir+=res[0]
                    print(res[0],neg)
                    pred.clear()
                    print("Zbir je: ", zbir)
        if cv2.waitKey(1) == 27:
            break
cap.release()