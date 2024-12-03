import os
from torch.utils.data import Dataset
from PIL import Image


class ImageFolder(Dataset):
    """ImageFolder Dataset for loading images organized in a directory structure.

    Args:
        root (str): Root directory containing class subdirectories.
        transform (callable, optional): A function/transform to apply to the images.
    """

    def __init__(self, root: str, transform=None) -> None:
        self.transform = transform
        self.samples = self._make_dataset(root)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = self._load_image(path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        """Loads an image from the given path."""
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    @staticmethod
    def _make_dataset(directory: str):
        """Creates a dataset of image paths and corresponding labels."""
        class_names = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        instances = []
        for class_name, class_index in class_to_idx.items():
            class_dir = os.path.join(directory, class_name)

            for root, _, file_names in os.walk(class_dir, followlinks=True):
                for file_name in sorted(file_names):
                    path = os.path.join(root, file_name)
                    if os.path.splitext(path)[1].lower() in {".jpg", ".jpeg", ".png"}:
                        instances.append((path, class_index))

        return instances
