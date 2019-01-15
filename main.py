from till.preprocessing import preprocessing
from till.preprocessing import face
from till.utils import create_folder, delete_folder

import time

# pre = preprocessing({"capture_rate": 0.2})
delete_folder("output")
create_folder("output")
# preprocessing.to_frames("Martin.mp4", "output", True)

detection = face.detection({"image_path": "./dnn.jpg", "method": "hog"})
print(detection.run_detection())

detection.selec_method("haar")
print(detection.run_detection())

detection.selec_method("dnn")
print(detection.run_detection())
