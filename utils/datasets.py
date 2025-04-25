import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from . import video_transforms

# Reference: https://github.com/NJU-PCALab/OpenVid-1M/blob/main/openvid/datasets/datasets.py

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def get_transforms_video(resolution=256):
    transform_video = transforms.Compose(
        [
            video_transforms.ToTensorVideo(),
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(resolution),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    return transform_video


def get_transforms_image(image_size=256):
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    return transform


class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        video_dir,
        num_frames=16,
        frame_interval=1,
        transform=None,
        max_samples=None, # control loaded video numbers
    ):
        video_paths = []
        for fname in sorted(os.listdir(video_dir)):
            if fname.endswith((".mp4", ".avi", ".mov")):
                full_path = os.path.join(video_dir, fname)
                video_paths.append(full_path)

        if max_samples is not None:
            video_paths = video_paths[:max_samples]

        self.samples = video_paths
        self.is_video = True
        self.transform = transform
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)

    def getitem(self, index):
        path = self.samples[index]

        is_exit = os.path.exists(path)
        if is_exit:
            vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
            total_frames = len(vframes)
        else:
            total_frames = 0

        loop_index = index
        while total_frames < self.num_frames or not is_exit:
            loop_index = (loop_index + 1) % len(self.samples)
            path = self.samples[loop_index]
            is_exit = os.path.exists(path)
            if is_exit:
                vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
                total_frames = len(vframes)
            else:
                total_frames = 0

        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= self.num_frames, f"{path} has not enough frames."
        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)

        video = vframes[frame_indice]
        video = self.transform(video)  # T C H W

        video = video.permute(1, 0, 2, 3)  # TCHW -> CTHW

        return {"video": video}

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    root = "" # path to video folder
    dataset = VideoDataset(
        video_dir=root,
        transform=get_transforms_video(),
        num_frames=16,
        frame_interval=3,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=1
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
