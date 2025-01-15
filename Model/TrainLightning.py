import os
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision.transforms import v2 as transforms, InterpolationMode
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from dotenv import load_dotenv
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score


photo_transforms = {
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
    ])}


class ToPytorchDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

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

    @classmethod
    def dataloader_from_hf(cls, dataset, transform, batch_size, shuffle):
        dataset_to_pytorch = cls(dataset, transform)
        dataloader = DataLoader(dataset_to_pytorch, batch_size=batch_size, shuffle=shuffle)
        return dataloader


class LightningModel(LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-4):
        super().__init__()
        self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.model.classifier[1].in_features, num_classes)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate

        self.accuracy = Accuracy(num_classes=num_classes, average='macro', task="binary")
        self.precision = Precision(num_classes=num_classes, average='macro', task="binary")
        self.recall = Recall(num_classes=num_classes, average='macro', task="binary")
        self.f1 = F1Score(num_classes=num_classes, average='macro', task="binary")

        self.predictions = []
        self.targets = []

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
        predictions = torch.argmax(y_hat, dim=1)
        self.accuracy.update(predictions, y)
        self.precision.update(predictions, y)
        self.recall.update(predictions, y)
        self.f1.update(predictions, y)

        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(y.cpu().numpy())

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def validation_epoch_end(self):
        accuracy = self.accuracy.compute()
        recall = self.recall.compute()
        precision = self.precision.compute()
        f1_score = self.f1_score.compute()

        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        self.log('val_f1', f1_score, prog_bar=True)

        if len(self.targets) > 0 and len(self.predictions) > 0:
            predictions = torch.tensor(self.predictions)
            targets = torch.tensor(self.targets)
            t_stat, p_value = ttest_ind(predictions.numpy(), targets.numpy(), equal_var=False)
            self.log('val_p_value', p_value, prog_bar=True)

        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.predictions = []
        self.targets = []

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=0.7, patience=5)
        return (
            {
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler
            }
        )


if __name__ == "__main__":
    load_dotenv()
    dataset_unprocessed = load_dataset(os.getenv("DATASET_UNPROCESSED"))
    dataset_autoprocessed = load_dataset(os.getenv("DATASET_AUTOPROCESSED"))
    dataset_manualprocessed = load_dataset(os.getenv("DATASET_MANUALPROCESSED"))

    batch_size = 16

    train_dataloader = ToPytorchDataset.dataloader_from_hf(dataset=dataset_manualprocessed["train"],
                                                           transform=photo_transforms["train"],
                                                           batch_size=batch_size,
                                                           shuffle=True)

    val_dataloader = ToPytorchDataset.dataloader_from_hf(dataset=dataset_manualprocessed["test"],
                                                         transform=photo_transforms["test"],
                                                         batch_size=batch_size,
                                                         shuffle=False)

    test_dataloader = ToPytorchDataset.dataloader_from_hf(dataset=dataset_unprocessed["test"],
                                                          transform=photo_transforms["test"],
                                                          batch_size=batch_size,
                                                          shuffle=False)

    model = LightningModel(num_classes=2, learning_rate=1e-4)
    logger = TensorBoardLogger("logs/")
    device = "gpu" if torch.cuda.is_available() else "cpu"

    checkpoints = ModelCheckpoint(
        monitor="val_accuracy",
        mode="max",
        save_top_k=3,
        dirpath="checkpoints/",
        filename="model_{epoch}_{val_accuracy:.4f}.pth",
    )

    trainer = Trainer(
        callbacks=[checkpoints],
        max_epochs=20,
        accelerator=device,
        devices=1,
        logger=logger,
        log_every_n_steps=5,
    )

    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)
