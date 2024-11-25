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
        img =  Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)