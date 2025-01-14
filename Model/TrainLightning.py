import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split, TensorDataset
import pytorch_lightning as pl
from torchvision.transforms import v2 as transforms, InterpolationMode
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights, EfficientNet


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, batch_size=16):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size

        self.train_transform = transforms.Compose([
        transforms.ToImage(),

        transforms.Resize(800, interpolation=InterpolationMode.BICUBIC),

        transforms.RandomResizedCrop(480, scale=(0.8, 1.0)),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(degrees=15),

        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),

        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),

        transforms.ToDtype(torch.float32, scale=True),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform =transforms.Compose([
            transforms.ToImage(),

            transforms.Resize(600, interpolation=InterpolationMode.BICUBIC),

            transforms.CenterCrop(480),

            transforms.ToDtype(torch.float32, scale=True),

            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def prepare_data(self):
        load_dataset(self.dataset_name)

    def setup(self, stage=None):
        dataset = load_dataset(self.dataset_name)
        if stage == "fit" or stage is None:
            self.train_data = dataset["train"].map(
                self._apply_transforms(self.train_transform),
                batched=False
            )
            self.val_data = dataset["test"].map(
                self._apply_transforms(self.val_transform),
                batched=False
            )

    def _apply_transforms(self, transform):
        def transform_function(example):
            example["image"] = transform(example["image"])
            assert isinstance(example["image"], torch.Tensor), f"Image should be a tensor, got {type(example['image'])}"
            return example

        return transform_function

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

class LightningModel(pl.LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        weights = EfficientNet_V2_M_Weights.DEFAULT
        self.model = efficientnet_v2_m(weights=weights)

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        self.loss_fn = nn.BCELoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

if __name__ == "__main__":
    dataset_unprocessed = "Kucharek9/AirForce1_unprocessed"
    dataset_autoprocessed = "Kucharek9/AirForce1_autoProcessed"
    dataset_manualprocessed = "Kucharek9/AirForce1_manualProcessed"

    data_module = DataModule(dataset_name=dataset_unprocessed, batch_size=16)
    model = LightningModel(num_classes=2, learning_rate=1e-4)
    logger = TensorBoardLogger("logs/")

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='gpu',
        devices=1,
        logger=logger,
        log_every_n_steps=5
    )

    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, data_module)
