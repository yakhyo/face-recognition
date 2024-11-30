from torch.utils.data import Dataset
import torch
from PIL import Image
import shutil
import torch.utils.data as data
from PIL import Image, ImageFile
import os

ImageFile.LOAD_TRUNCATED_IAMGES = True


# https://github.com/pytorch/vision/issues/81

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return
    except IOError:
        print('Cannot load image ' + path)


class ImageFolder(Dataset):
    """ImageFolder Dataset"""

    def __init__(self, root: str, transform=None) -> None:

        self.transform = transform
        self.samples = self.make_dataset(root)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = self.load_image(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def load_image(path):
        with open(path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')

        return image

    @staticmethod
    def make_dataset(directory):
        class_names = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        instances = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)

            for root, _, file_names in sorted(os.walk(target_dir, followlinks=True)):
                for file_name in sorted(file_names):
                    path = os.path.join(root, file_name)
                    base, ext = os.path.splitext(path)
                    if ext.lower() in [".jpg", ".jpeg", ".png"]:
                        item = path, class_index
                        instances.append(item)

        return instances


class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform=None):
        self.root = root
        self.imgList = []
        with open(fileList, 'r') as file:
            for line in file.readlines():
                imgPath, label = line.strip().split(' ')
                self.imgList.append((imgPath, int(label[3:])))
        self.transform = transform

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        path = os.path.join("data/train/"+self.root, imgPath)
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)
