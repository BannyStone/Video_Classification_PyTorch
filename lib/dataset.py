import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint

class VideoRecord(object):
    def __init__(self, row, root_path):
        self._data = row
        self._root_path = root_path

    @property
    def path(self):
        return os.path.join(self._root_path, self._data[0])

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoDataSet(data.Dataset):
    def __init__(self, root_path, list_file, 
                 t_length=32, t_stride=2, num_segments=3, 
                 image_tmpl='img_{:05d}.jpg', 
                 transform=None, style="Dense", 
                 phase="Train"):
        """
        :style: Dense, for 2D and 3D model, and Sparse for TSN model
        :phase: Train, Val, Test
        """

        self.root_path = root_path
        self.list_file = list_file
        self.t_length = t_length
        self.t_stride = t_stride
        self.num_segments = num_segments
        self.image_tmpl = image_tmpl
        self.transform = transform
        assert(style == "Dense"), "Sparse is not supported yet."
        self.style = style
        self.phase = phase
        assert(t_length > 0), "Length of time must be bigger than zero."
        assert(t_stride > 0), "Stride of time must be bigger than zero."

        self._parse_list()

    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' '), self.root_path) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        offset = 0
        average_duration = record.num_frames - (self.t_length - 1) * self.t_stride - 1
        if average_duration >= 0:
            offset = randint(average_duration + 1)
        elif record.num_frames > self.t_length:
            while(self.t_stride - 1 > 0):
                self.t_stride -= 1
                average_duration = record.num_frames - (self.t_length - 1) * self.t_stride - 1
                if average_duration >= 0:
                    offset = randint(average_duration + 1)
                    break
        else:
            self.t_stride = 1
        return [offset + 1]

    def _get_val_indices(self, record):
        """
        get indices in val phase
        """
        valid_offset_range = record.num_frames - (self.t_length - 1) * self.t_stride - 1
        offset = int(valid_offset_range / 2.0)
        if offset < 0:
            offset = 0
        return [offset + 1]

    def _get_test_indices(self, record):
        """
        get indices in test phase
        """
        valid_offset_range = record.num_frames - (self.t_length - 1) * self.t_stride - 1
        interval = valid_offset_range / (self.num_segments - 1)
        offsets = []
        for i in range(self.num_segments):
            offset = int(i * interval)
            if offset > valid_offset_range:
                offset = valid_offset_range
            if offset < 0:
                offset = 0
            offsets.append(offset)
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]

        if self.phase == "Train":
            indices = self._sample_indices(record)
        elif self.phase == "Val":
            indices = self._get_val_indices(record)
        elif self.phase == "Test":
            indices = self._get_test_indices(record)
        else:
            raise TypeError("Unsuported phase {}".format(self.phase))

        return self.get(record, indices)

    def get(self, record, indices):

        images = list()
        for ind in indices:
            p = int(ind)
            for i in range(self.t_length):
                ptr = p + i * self.t_stride
                if ptr <= record.num_frames:
                    imgs = self._load_image(record.path, ptr)
                else:
                    imgs = self._load_image(record.path, record.num_frames)
                images.extend(imgs)

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
