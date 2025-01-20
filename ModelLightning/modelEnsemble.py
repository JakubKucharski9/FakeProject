import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from scipy.stats import ttest_ind
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

class LightningModelEnsemble(LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-4, freeze_bn=False, model_list=None, mode=None):
        super().__init__()
        self.models = model_list
        self.mode = mode
        self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        self.num_classes = num_classes

        if freeze_bn:
            for param in self.model.parameters():
                param.requires_grad = False
            self.unfreeze_backbone()

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.model.classifier[1].in_features, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.accuracy = Accuracy(num_classes=num_classes, average='macro', task="multiclass")
        self.precision = Precision(num_classes=num_classes, average='macro', task="multiclass")
        self.recall = Recall(num_classes=num_classes, average='macro', task="multiclass")
        self.f1 = F1Score(num_classes=num_classes, average='macro', task="multiclass")

        self.predictions = []
        self.targets = []

    def forward(self, x):
        if self.models is None and self.mode is None:
            return self.model(x)
        else:
            predictions = []
            for model in self.models:
                model = model.to(self.device)
                with torch.no_grad():
                    predictions.append(model(x))

            if self.mode == "soft_voting":
                output = torch.mean(torch.stack(predictions), dim=0)
            elif self.mode == "hard_voting":
                votes = torch.argmax(torch.stack(predictions), dim=2)
                votes = votes.clamp(0, self.num_classes - 1)
                output = nn.functional.one_hot(votes, num_classes=self.num_classes).sum(dim=0)
            else:
                raise ValueError(f"Unknown mode {self.mode}")

            return output

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.float()
        y = y.to(self.device)
        x_hat = self(x)
        loss = self.loss_fn(x_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = y.float()
        y = y.to(self.device)

        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        predictions = torch.argmax(y_hat, dim=1)
        self.accuracy.update(predictions, y)
        self.precision.update(predictions, y)
        self.recall.update(predictions, y)
        self.f1.update(predictions, y)

        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(y.cpu().numpy())

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log('val_accuracy', self.accuracy.compute(), prog_bar=True)
        self.log('val_precision', self.precision.compute(), prog_bar=True)
        self.log('val_recall', self.recall.compute(), prog_bar=True)
        self.log('val_f1', self.f1.compute(), prog_bar=True)

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
        loss = self.loss_fn(y_hat, y)
        predictions = torch.argmax(y_hat, dim=1)
        accuracy = self.accuracy(predictions, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_accuracy', accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.7, patience=5)
        return (
            {
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
                'monitor': 'val_accuracy',
            }
        )

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True
