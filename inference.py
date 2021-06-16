
import keras
import numpy as np
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import array_to_img, img_to_array

unique_labels = ["Parasite", "Uninfected"]
model = keras.models.load_model("model_vgg19.h5")

def predict(img_path):   #mandatory: function name should be predict and it accepts a string which is image location
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image/255
        image=np.expand_dims(image,axis=0)
        y_pred = model.predict(image)
        y_pred = np.argmax(y_pred, axis=1)
        predicted_categories = [unique_labels[i] for i in y_pred]

        return predicted_categories #mandatory: the return should be a string
