"""Preprocessing"""
import cv2
from datetime import datetime
import numpy as np
import math
from sklearn.model_selection import train_test_split
import shutil
from till.utils import (get_from_dict, loader, str_join,
                        crop_Image_rect, save_Image, get_folder_info, str_join_array, create_folder, create_multiple_folder, rev_str_join_array)
from till.preprocessing import face


class preprocessing(object):
    def __init__(self, args: {}):
        self.capture_rate = get_from_dict(args, ["capture_rate"])

    @staticmethod
    def to_frames(video_path: str = None, output_path: str = None, verboose: bool = True):
        video = cv2.VideoCapture(video_path)
        len_frames = preprocessing.number_frames(video)
        success, frame = video.read()
        count = 0

        while success:
            if verboose:
                loader(count, len_frames, "extracting frames from %s " %
                       (video_path))

            cv2.imwrite(
                str_join([output_path, str(datetime.now())+'.jpg']), frame)
            success, frame = video.read()
            count += 1

    @staticmethod
    def number_frames(video):
        return int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    @staticmethod
    def to_face(image_folder, output_path, method: str = "dnn", size=(299, 299)):
        face_detection = face.detection({"image_path": "", "method": method})
        print(image_folder)
        face_detection.set_image(image_folder)
        boxex = face_detection.run_detection()
        count = 1
        for box in boxex:
            img_crop = crop_Image_rect(image_folder, box=box, size=size)
            output_path = (output_path.split(".jpg"))[0]
            print("SAVE IN " + output_path+str(count)+".jpg")
            save_Image(img_crop, output_path+str(count)+".jpg")
            count += 1

    @staticmethod
    def flow_from_face_directory(directory, output, method="dnn"):
        (folders, _) = get_folder_info(directory)
        folders_path = str_join_array(directory, folders)
        folders_output_path = str_join_array(output, folders)
        count = 0
        for folder in folders_path:
            create_folder(folders_output_path[count])
            (_, image_in_folder) = get_folder_info(folder)
            path_image_in_folder = str_join_array(folder, image_in_folder)
            path_output_image_in_folder = str_join_array(
                folders_output_path[count], image_in_folder)

            print(path_image_in_folder)
            for i in range(len(path_image_in_folder)):
                preprocessing.to_face(
                    path_image_in_folder[i], path_output_image_in_folder[i])
            count += 1

    @staticmethod
    def fff_directory(directory, output, method="dnn"):
        preprocessing.flow_from_face_directory(directory, output, method)

    @staticmethod
    def flow_from_video_directory(directory, output, verboose=True):
        (folders, _) = get_folder_info(directory)
        folders_path = str_join_array(directory, folders)
        folders_output_path = str_join_array(output, folders)
        count = 0
        for folder in folders_path:
            create_folder(folders_output_path[count])
            (_, videos_in_folder) = get_folder_info(folder)
            path_video_in_folder = str_join_array(folder, videos_in_folder)

            for i in range(len(path_video_in_folder)):
                preprocessing.to_frames(
                    path_video_in_folder[i], folders_output_path[count], verboose)
            count += 1

    @staticmethod
    def ffv_directory(directory, output, verboose=True):
        preprocessing.flow_from_video_directory(directory, output, verboose)

    @staticmethod
    def prepare_dataset(input="videos", output="data", faces="faces", verboose=True):
        create_multiple_folder([input, output, faces])
        preprocessing.ffv_directory(input, output)
        preprocessing.fff_directory(output, faces)

    @staticmethod
    def generate_dataset(dataset, faces):
        sets = ["train", "val", "test"]
        folders_set = str_join_array(dataset, sets)
        create_multiple_folder(folders_set)
        (labels, _) = get_folder_info(faces)
        for label in labels:
            to_folder = rev_str_join_array(label, folders_set)
            print(to_folder)
            create_multiple_folder(to_folder)
            label_origen = str_join([faces, label])
            (_, label_images) = get_folder_info(label_origen)
            label_images = np.array(label_images)
            np.random.shuffle(label_images)

            # split image 70% train - 30% test
            img_train_, img_test = label_images[:(math.floor(len(
                label_images)*0.7))], label_images[(math.floor(len(label_images)*0.7)):]
            # From the train set, generate the validation set
            img_train, img_val = train_test_split(
                img_train_, test_size=0.3)

            img_train = str_join_array(label_origen, img_train)
            img_test = str_join_array(label_origen, img_test)
            img_val = str_join_array(label_origen, img_val)

            for x in img_train:
                shutil.copy2(x, to_folder[0])

            for y in img_val:
                shutil.copy2(y, to_folder[1])

            for z in img_test:
                shutil.copy2(z, to_folder[2])
