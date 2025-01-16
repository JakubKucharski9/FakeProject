import torch
from torchvision.transforms import v2 as transforms, InterpolationMode


def photo_transforms():
    return {
        "train": transforms.Compose([
            transforms.ToImage(),

            transforms.Resize(800, interpolation=InterpolationMode.BICUBIC),

            transforms.RandomResizedCrop(480, scale=(0.8, 1.0)),

            transforms.RandomHorizontalFlip(p=0.5),

            transforms.RandomRotation(degrees=15),

            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),

            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),

            transforms.ToDtype(torch.float32, scale=True),

            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ]),
        "test": transforms.Compose([
            transforms.ToImage(),

            transforms.Resize(600, interpolation=InterpolationMode.BICUBIC),

            transforms.CenterCrop(480),

            transforms.ToDtype(torch.float32, scale=True),

            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }