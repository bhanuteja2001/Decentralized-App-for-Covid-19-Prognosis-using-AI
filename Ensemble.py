import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import os


MODEL_PTH = os.getcwd()
model_check = load_model(MODEL_PTH+"/weights/VGG16_CL.h5")
model = load_model(MODEL_PTH+"/weights/Dense_net.h5")
model_2 = load_model(MODEL_PTH+"/weights/incpetion_v3.h5")

class Ensemble:

    #Image Path - absolute path of the Input Image
    
    def Result(self, img_array):
        #plt.imshow(img_array)
        #plt.show()
        

        new_img_array = cv2.resize(img_array, (224, 224))
        #new_img_array = img_array.resize((128,128))
        dt = []
        dt.append(new_img_array)
        X = np.array(dt)
        X = X/255
        val = model_check.predict(X)
        Rs = np.argmax(val,axis=1)[0]
        print("Prediction L:: ",Rs, max(val[0]))

        if Rs == 0 and max(val[0]) > 0.75:
            #224 DenseNet
            new_img_array = cv2.resize(img_array, (224, 224))
            #new_img_array = img_array.resize((128,128))
            dt = []
            dt.append(new_img_array)
            X = np.array(dt)
            X = X/255
            val = model.predict(X)

            #256 Inception V3
            new_img_array_2 = cv2.resize(img_array, (224, 224))
            #new_img_array_2 = img_array.resize((224, 224))
            dt_2 = []
            dt_2.append(new_img_array_2)
            X_2 = np.array(dt_2)
            X_2 = X_2/255
            val_2 = model_2.predict(X_2)

            preds = [val, val_2]
            weights = [0.5,0.5]
            weighted_preds = np.tensordot(preds, weights, axes=((0),(0)))

            
            return str(np.argmax(weighted_preds,axis=1))
        else:
            return str("Please upload a correct Image")

