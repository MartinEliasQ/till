"""Preprocessing"""
import cv2
from datetime import datetime
from till.utils import (get_from_dict, loader, str_join)


class preprocessing(object):
    def __init__(self, args: {}):
        self.capture_rate = get_from_dict(args, ["capture_rate"])

    @staticmethod
    def to_frames(video_path: str = None, output_path: str = None, verboose: bool = False):
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
