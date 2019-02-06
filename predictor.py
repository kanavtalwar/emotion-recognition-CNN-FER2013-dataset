from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

class Predictor:
    def __init__(self):
        self.model = load_model('./model/saved_models/emotion1543161952.3752954.h5')
        self.graph = tf.get_default_graph()
        self.img_height = 48
        self.img_width = 48
        self.emotion_dict = {
            0: "Angry", 
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral"
        }

    def predict(self, image):
        image = np.expand_dims(np.expand_dims(cv2.resize(image, (self.img_height, self.img_width)), -1), 0).astype('float32')
        image /= 255.0

        with self.graph.as_default():
            return self.emotion_dict[np.argmax(self.model.predict(image))]

# if __name__ == "__main__":
#     img = np.array(Image.open('./data/test-images/happy.jpeg')).astype('float32')
#     predictor = Predictor()

#     print(predictor.predict(img))
