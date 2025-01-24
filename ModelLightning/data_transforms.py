from nike_pack import *


def photo_transforms():
    return {
        "train": transforms.Compose([
            transforms.ToImage(),

            transforms.Resize(600, interpolation=InterpolationMode.BICUBIC),

            transforms.RandomResizedCrop(480, scale=(0.8, 1.0)),

            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),

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