"""ImageNet data loading shared by training and frozen evaluation."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def evaluation_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FlatValidation(Dataset):
    """Flat val/ images with zero-based labels in meta/val.txt."""

    def __init__(self, root, transform, num_classes):
        root = Path(root)
        self.directory = root / "val"
        self.transform = transform
        self.samples = []
        for line in (root / "meta" / "val.txt").read_text().splitlines():
            filename, label = line.split()
            label = int(label)
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Validation filenames must be relative to val/")
            if not 0 <= label < num_classes:
                raise ValueError("Validation labels must be zero-based ImageFolder class indices")
            if not (self.directory / relative).is_file():
                raise FileNotFoundError(self.directory / relative)
            self.samples.append((filename, label))
        if not self.samples:
            raise ValueError("Empty validation manifest")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename, label = self.samples[index]
        with Image.open(self.directory / filename) as image:
            return self.transform(image.convert("RGB")), label


def evaluation_datasets(root):
    root = Path(root)
    transform = evaluation_transform()
    train = datasets.ImageFolder(root / "train", transform=transform)
    if (root / "meta" / "val.txt").is_file():
        validation = FlatValidation(root, transform, len(train.classes))
    else:
        validation = datasets.ImageFolder(root / "val", transform=transform)
        if train.class_to_idx != validation.class_to_idx:
            raise ValueError("Training and validation class mappings differ")
    return train, validation
