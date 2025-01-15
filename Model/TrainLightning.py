import os
import torch
import torch.nn as nn
from PIL.Image import Image
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from Train import photo_transforms
from dotenv import load_dotenv


class ToPytorchDataset(Dataset):
    def __init__(self, dataset_from_huggingface, transform=None):
        self.dataset_from_huggingface = dataset_from_huggingface
        self.transform = transform

    def __len__(self):
        return len(self.dataset_from_huggingface)

    def __getitem__(self, idx):
        item = self.dataset_from_huggingface[idx]

        image = item["image"]
        label = item["label"]

        # Convert image to RGB to delete alpha channel
        if image.mode == "RGBA":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_one_hot = torch.zeros(2)
        label_one_hot[label] = 1.0

        return image, label_one_hot


class LightningModel(LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-4):
        super().__init__()
        self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x_hat = model(x)
        loss = self.loss_fn(x_hat.squeeze(), y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat.squeeze(), y.float())
        predictions = (torch.sigmoid(y_hat) > 0.5).float()
        accuracy = (predictions == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat.squeeze(), y.float())
        predictions = (torch.sigmoid(y_hat) > 0.5).float()
        accuracy = (predictions == y).float().mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_accuracy', accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    load_dotenv()
    dataset_unprocessed = load_dataset(os.getenv("DATASET_UNPROCESSED"))
    dataset_autoprocessed = load_dataset(os.getenv("DATASET_AUTOPROCESSED"))
    dataset_manualprocessed = load_dataset(os.getenv("DATASET_MANUALPROCESSED"))

    batch_size = 16

    dataset_train_loader_to_pytorch = ToPytorchDataset(dataset_manualprocessed["train"], transform=photo_transforms["train"])
    train_dataloader = DataLoader(dataset_train_loader_to_pytorch, batch_size=batch_size, shuffle=True)

    dataset_val_loader_to_pytorch = ToPytorchDataset(dataset_manualprocessed["test"], transform=photo_transforms["test"])
    val_dataloader = DataLoader(dataset_val_loader_to_pytorch, batch_size=batch_size, shuffle=False)

    dataset_test_loader_to_pytorch = ToPytorchDataset(dataset_unprocessed["test"], transform=photo_transforms["test"])
    test_dataloader = DataLoader(dataset_test_loader_to_pytorch, batch_size=batch_size, shuffle=False)

    model = LightningModel(num_classes=2, learning_rate=1e-4)
    logger = TensorBoardLogger("logs/")
    device = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        max_epochs=20,
        accelerator=device,
        devices=1,
        logger=logger,
        log_every_n_steps=5,
    )

    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)
