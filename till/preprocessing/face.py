"""face"""
import cv2
import dlib
import numpy as np
from PIL import Image
from till.utils import (get_from_dict, read_image,
                        convert_to_blob, read_net_dnn)


class detection(object):
    def __init__(self, args: {}):
        print(args)
        self.image_path = get_from_dict(args, ["image_path"])[0]
        print(self.image_path)
        self.image = read_image(self.image_path)
        self.selec_method(get_from_dict(args, ["method"])[0])

    def hog(self):
        print("HOG")
        face_detector = dlib.get_frontal_face_detector()
        detected_faces = face_detector(self.image, 1)
        boxes = list()
        for _, face_box in enumerate(detected_faces):
            x_size = face_box.left()
            y_size = face_box.top()
            w_size = face_box.right()
            h_size = face_box.bottom()
            box = (x_size, y_size, w_size, h_size)
            boxes.append(box)
        return box

    def haar(self, cascade_xml: str = "src/haarcascade_frontalface_alt2.xml"):
        print("Haar")
        face_detector = cv2.CascadeClassifier(cascade_xml)
        detected_faces = face_detector.detectMultiScale(self.image, 1.3, 5)
        boxes = list()
        for (x, y, w, h) in detected_faces:
            ima = Image.open(self.image_path)
            center_x = x + w/2
            center_y = y + h/2
            b_dim = min(max(w, h)*1.2, ima.width, ima.height)
            box = ((center_x-b_dim/2), (center_y-b_dim/2),
                   (center_x+b_dim/2), (center_y+b_dim/2))
            boxes.append(box)
        return boxes

    def dnn(self):
        print("DNN")
        (h, w) = self.image.shape[:2]
        blob = convert_to_blob(self.image)
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        boxes = list()
        for i in range(0, detections.shape[2]):
            confidense = detections[0, 0, i, 2]
            if confidense > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                boxes.append(box.astype("int"))
        return boxes

    def run_detection(self):
        return self.method()

    def selec_method(self, method: str):
        if method == "dnn":
            print("[INFO] loading from model (DNN)...")
            self.dnn_net = read_net_dnn()
        self.method = {
            "haar": lambda: self.haar(),
            "hog": lambda: self.hog(),
            "dnn": lambda: self.dnn()
        }[method]
