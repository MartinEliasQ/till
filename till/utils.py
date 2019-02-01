"""utils"""
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import shutil
from sklearn.metrics import confusion_matrix
import sys
import time


def str_join(paths: []):
    """
    Concatenate a list of paths
    Attributes:
        paths: List of paths.
    """
    return "/".join(paths)


def str_join_array(label, array):
    return np.array(list(map(lambda x: str_join([label, x]), array)))


def rev_str_join_array(label, array):
    return np.array(list(map(lambda x: str_join([x, label]), array)))


def create_folder(path: str):
    """
    Create folder in the path
    Attributes:
        paths: Path where will create the folder.
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except:
        print("An error occured.")


def delete_folder(path: str):
    """
    Create folder in the path
    Attributes:
        paths: Path where will create the folder.
    """
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
        return True
    except:
        print("An error occured.")


def get_folder_info(path: str):
    """Get a list with the elements of a folder
    Attributes:
        path: Folder with the elements
        (folders in path, elements in path)
    """
    try:
        folder_list = next(os.walk(path))
        return (folder_list[1], folder_list[2])
    except TypeError:
        print(TypeError)


def create_multiple_folder(paths: []):
    for path in paths:
        create_folder(path)
    return True


def delete_multiple_folder(paths: []):
    for path in paths:
        delete_folder(path)
    return True


def get_from_dict(values: str, elements: []):
    return [values[key] for key in (elements)]


def verboose(message: str):
    print(message)


def loader(count, total, status=''):
    # From https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.1 * count / float(total), 1)  # 100.1 to display%
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('\r[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)


def read_image(image_path: str):
    return cv2.imread(image_path)


def read_image2(image_path: str):
    return Image.open(image_path)


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_to_blob(image):
    return cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (103.93, 116.77, 123.68))


def read_net_dnn(prototxt: str = "src/deploy.prototxt", model: str = "src/res10_300x300_ssd_iter_140000.caffemodel"):
    return cv2.dnn.readNetFromCaffe(prototxt, model)


def crop_Image_rect(image, box: np.array, size):
    image = read_image2(image)
    return image.crop(box).resize(size)


def save_Image(image, path):
    try:
        image.save(path)
        return True
    except:
        return False


def accuracy(tn, fp, fn, tp):
    return (tn + tp)/(tn + fp + fn + tp)


def sensitivity(tn, fp, fn, tp):
    '''True positive rate || recall Sensitivity'''
    return (tp)/(fn+tp)


def specificity(tn, fp, fn, tp):
    '''True negative rate'''
    return (tn)/(tn + fp)


def precision(tn, fp, fn, tp):
    return (tp)/(fp + tp)


def metrics(Y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(Y_true, y_pred).ravel()
    print(Y_true, y_pred)
    metrics = {"accuracy": accuracy(tn, fp, fn, tp),
               "sensitivity": sensitivity(tn, fp, fn, tp),
               "specificity": specificity(tn, fp, fn, tp),
               "precision": precision(tn, fp, fn, tp)}

    return metrics


def convert_to_np_array(element):
    return np.array(element)


def load_image_and_convert_np(element):
    img = read_image2(element)
    return convert_to_np_array(img)
