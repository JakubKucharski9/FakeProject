from torch.utils.data import DataLoader, Dataset

class ToPytorchDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        label = self.dataset[idx]["label"]

        if image.mode == "RGBA":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    @classmethod
    def dataloader_from_hf(cls, dataset, transform, batch_size, shuffle, num_workers, persistent_workers=None):
        dataset_to_pytorch = cls(dataset, transform)
        dataloader = DataLoader(dataset_to_pytorch, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, persistent_workers=persistent_workers)
        return dataloader
