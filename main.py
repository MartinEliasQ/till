from till.preprocessing import preprocessing
from till.preprocessing import face
from till.transfer import tl
from till.utils import create_folder, delete_folder, get_folder_info, str_join, str_join_array, rev_str_join_array
import numpy as np
import time

#
#  pre = preprocessing({"capture_rate": 0.2})
# delete_folder("output")
# create_folder("output")
# print(get_folder_info("./till/tests"))
#preprocessing.to_frames("videos/rubias/demo.mp4", "output", True)
#preprocessing.to_frames("videos/demo.mp4", "output", True)
# preprocessing.to_face(["hog.jpg", "hog.jpg", "hog.jpg"],
#                      "output/img", method="hog")
#preprocessing.flow_from_video_directory("videos", "data")
#preprocessing.to_face(["dnn.jpg"], "output/img", method="haar")
#preprocessing.to_face(["dnn.jpg"], "output/img", method="dnn")
#detection = face.detection({"image_path": "dnn.jpg", "method": "hog"})

# print(detection.run_detection())

# detection.selec_method("haar")
# print(detection.run_detection())
# detection.set_image("haar.jpg")
# detection.selec_method("dnn")
# print(detection.run_detection())
#preprocessing.flow_from_face_directory("data", "output")
preprocessing.prepare_dataset()

preprocessing.generate_dataset("dataset", "faces")

# print(tl.check_device())
# print(tl.transforms_data())
