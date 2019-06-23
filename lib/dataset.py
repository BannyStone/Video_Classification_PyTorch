import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint

import torch

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

class VideoDebugDataSet(data.Dataset):
    """
    """
    def __len__(self):
        return 100

    def __getitem__(self, index):
        np.random.seed(12345)
        input_tensor = (np.random.random_sample((3,18,224,224)) - 0.5) * 2
        return torch.from_numpy(input_tensor).to(torch.float), 0

class VideoDataSet(data.Dataset):
    def __init__(self, root_path, list_file, 
                 t_length=32, t_stride=2, num_segments=1, 
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
        assert(style in ("Dense", "UnevenDense")), "Only support Dense and UnevenDense"
        self.style = style
        self.phase = phase
        assert(t_length > 0), "Length of time must be bigger than zero."
        assert(t_stride > 0), "Stride of time must be bigger than zero."

        self._parse_list()

    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' '), self.root_path) for x in open(self.list_file)]
        # self.video_list = [VideoRecord(x.strip().split(' '), self.root_path) for x in open(self.list_file) if VideoRecord(x.strip().split(' '), self.root_path).num_frames > 240]
        # print(len(self.video_list))

    @staticmethod
    def dense_sampler(num_frames, length, stride=1):
        t_length = length
        t_stride = stride
        # compute offsets
        offset = 0
        average_duration = num_frames - (t_length - 1) * t_stride - 1
        if average_duration >= 0:
            offset = randint(average_duration + 1)
        elif num_frames > t_length:
            while(t_stride - 1 > 0):
                t_stride -= 1
                average_duration = num_frames - (t_length - 1) * t_stride - 1
                if average_duration >= 0:
                    offset = randint(average_duration + 1)
                    break
            assert(t_stride >= 1), "temporal stride must be bigger than zero."
        else:
            t_stride = 1
        # sampling
        samples = []
        for i in range(t_length):
            samples.append(offset + i * t_stride + 1)
        return samples

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        if self.style == "Dense":
            frames = []
            average_duration = record.num_frames / self.num_segments
            offsets = [average_duration * i for i in range(self.num_segments)]
            for i in range(self.num_segments):
                samples = self.dense_sampler(average_duration, self.t_length, self.t_stride)
                samples = [sample + offsets[i] for sample in samples]
                frames.extend(samples)
            return {"dense": frames}
        elif self.style == "UnevenDense":
            sparse_frames = []
            average_duration = record.num_frames / self.num_segments
            offsets = [average_duration * i for i in range(self.num_segments)]
            dense_frames = self.dense_sampler(record.num_frames, self.t_length, self.t_stride)
            dense_seg = -1
            for i in range(self.num_segments):
                if dense_frames[self.t_length//2] >= offsets[self.num_segments - i - 1]:
                    dense_seg = self.num_segments - i - 1
                    break
                else:
                    continue
            assert(dense_seg != -1)
            # dense_seg = randint(self.num_segments)
            for i in range(self.num_segments):
                # if i == dense_seg:
                    # samples = self.dense_sampler(average_duration, self.t_length, self.t_stride)
                    # samples = [sample + offsets[i] for sample in samples]
                    # dense_frames.extend(samples)
                    # dense_seg = -1 # set dense seg to -1 and check after sampling.
                if i != dense_seg:
                    samples = self.dense_sampler(average_duration, 1)
                    samples = [sample + offsets[i] for sample in samples]
                    sparse_frames.extend(samples)
            return {"dense":dense_frames, "sparse":sparse_frames}
        else:
            return

    def _get_val_indices(self, record):
        """
        get indices in val phase
        """
        # valid_offset_range = record.num_frames - (self.t_length - 1) * self.t_stride - 1
        valid_offset_range = record.num_frames - (self.t_length - 1) * self.t_stride - 1
        offset = int(valid_offset_range / 2.0)
        if offset < 0:
            offset = 0
        samples = []
        for i in range(self.t_length):
            samples.append(offset + i * self.t_stride + 1)
        return {"dense": samples}

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
            offsets.append(offset + 1)
        frames = []
        for i in range(self.num_segments):
            for j in range(self.t_length):
                frames.append(offsets[i] + j*self.t_stride)
                # frames.append(offsets[i]+j)
        return {"dense": frames}

    def __getitem__(self, index):
        record = self.video_list[index]

        if self.phase == "Train":
            indices = self._sample_indices(record)
            return self.get(record, indices, self.phase)
        elif self.phase == "Val":
            indices = self._get_val_indices(record)
            return self.get(record, indices, self.phase)
        elif self.phase == "Test":
            indices = self._get_test_indices(record)
            return self.get(record, indices, self.phase)
        else:
            raise TypeError("Unsuported phase {}".format(self.phase))

    def get(self, record, indices, phase):
        # dense process data
        def dense_process_data():
            images = list()
            for ind in indices['dense']:
                ptr = int(ind)
                if ptr <= record.num_frames:
                    imgs = self._load_image(record.path, ptr)
                else:
                    imgs = self._load_image(record.path, record.num_frames)
                images.extend(imgs)
            return self.transform(images)
        # unevendense process data
        def unevendense_process_data():
            dense_images = list()
            sparse_images = list()
            for ind in indices['dense']:
                ptr = int(ind)
                if ptr <= record.num_frames:
                    imgs = self._load_image(record.path, ptr)
                else:
                    imgs = self._load_image(record.path, record.num_frames)
                dense_images.extend(imgs)
            for ind in indices['sparse']:
                ptr = int(ind)
                if ptr <= record.num_frames:
                    imgs = self._load_image(record.path, ptr)
                else:
                    imgs = self._load_image(record.path, record.num_frames)
                sparse_images.extend(imgs)

            images = dense_images + sparse_images
            return self.transform(images)
        if phase == "Train":
            if self.style == "Dense":
                process_data = dense_process_data()
            elif self.style == "UnevenDense":
                process_data = unevendense_process_data()
        elif phase in ("Val", "Test"):
            process_data = dense_process_data()
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

class ShortVideoDataSet(VideoDataSet):
    def __init__(self, root_path, list_file, 
                 t_length=32, t_stride=2, num_segments=1, 
                 image_tmpl='img_{:05d}.jpg', 
                 transform=None, style="Dense", 
                 phase="Train"):
        """
        :style: Dense, for 2D and 3D model, and Sparse for TSN model
        :phase: Train, Val, Test
        """

        super(ShortVideoDataSet, self).__init__(root_path, 
            list_file, t_length, t_stride, num_segments, 
            image_tmpl, transform, style, phase)


    def _get_val_indices(self, record):
        """
        get indices in val phase
        """
        # valid_offset_range = record.num_frames - (self.t_length - 1) * self.t_stride - 1
        t_stride = self.t_stride
        valid_offset_range = record.num_frames - (self.t_length - 1) * t_stride - 1
        offset = int(valid_offset_range / 2.0)

        if record.num_frames > self.t_length:
            while(offset < 0 and t_stride > 1):
                t_stride -= 1
                valid_offset_range = record.num_frames - (self.t_length - 1) * t_stride - 1
                offset = int(valid_offset_range / 2.0)
        else:
            t_stride = 1
            valid_offset_range = record.num_frames - (self.t_length - 1) * t_stride - 1
            offset = int(valid_offset_range / 2.0)

        if offset < 0:
            offset = 0
        samples = []
        for i in range(self.t_length):
            samples.append(offset + i * t_stride + 1)
        return {"dense": samples}

    def _get_test_indices(self, record):
        """
        get indices in test phase
        """
        t_stride = self.t_stride
        valid_offset_range = record.num_frames - (self.t_length - 1) * t_stride - 1
        while(valid_offset_range < (self.num_segments - 1) and t_stride > 1):
            t_stride -= 1
            valid_offset_range = record.num_frames - (self.t_length - 1) * t_stride - 1
        if valid_offset_range < 0:
            valid_offset_range = 0
        interval = valid_offset_range / (self.num_segments - 1)
        offsets = []
        for i in range(self.num_segments):
            offset = int(i * interval)
            if offset > valid_offset_range+1:
                offset = valid_offset_range+1
            if offset < 0:
                offset = 0
            offsets.append(offset + 1)
        frames = []
        for i in range(self.num_segments):
            for j in range(self.t_length):
                frames.append(offsets[i] + j * t_stride)
                # frames.append(offsets[i]+j)
        return {"dense": frames}


if __name__ == "__main__":
    td = VideoDataSet(root_path="../data/kinetics400/access/kinetics_train_rgb_img_256_340/",
                                 list_file="../data/kinetics400/kinetics_train_list.txt",
                                 t_length=16,
                                 t_stride=4,
                                 num_segments=3,
                                 image_tmpl="image_{:06d}.jpg",
                                 style="UnevenDense",
                                 phase="Train")
    # sample0 = td[0]
    import pdb
    pdb.set_trace()
