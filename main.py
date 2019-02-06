from face_detect.face_detect import Face
from predictor import Predictor

if __name__ == "__main__":
    face_tracker = Face()
    model = Predictor()

    face_tracker.WebCamFaceDetect(model)
    