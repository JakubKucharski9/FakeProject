from torchmetrics import ConfusionMatrix
import seaborn as sns
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights, efficientnet_b5, EfficientNet_B5_Weights

from nike_pack import *


class LightningModel(LightningModule):
    def __init__(self, num_classes, learning_rate, weight_decay, threshold=0.5, dropout=0.5, scheduler_factor=0.1, scheduler_patience=5, model_to_use="efficientnetv2"):
        super().__init__()
        self.save_hyperparameters()
        if model_to_use == 'efficientnetv2':
            self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.model.classifier[1].in_features, num_classes)
            )
        elif model_to_use == 'regnet':
            self.model = regnet_y_8gf(weights=RegNet_Y_8GF_Weights.DEFAULT)
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.model.fc.in_features, num_classes)
            )
        elif model_to_use == 'efficientnetb4':
            self.model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.model.classifier[1].in_features, num_classes)
            )
        elif model_to_use == 'efficientnetb5':
            self.model = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.model.classifier[1].in_features, num_classes)
            )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.threshold = threshold
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience

        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")
        self.confusion_matrix = ConfusionMatrix(task="binary", num_classes=2)

    def forward(self, x):
        return self.model(x).squeeze(dim=1)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.float())

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.float())
        predictions = (torch.sigmoid(y_hat) > self.threshold).int()
        y = y.int()

        self.accuracy.update(predictions, y)
        self.precision.update(predictions, y)
        self.recall.update(predictions, y)
        self.f1.update(predictions, y)
        self.confusion_matrix.update(predictions, y)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def on_validation_epoch_end(self):
        cm = self.confusion_matrix.compute().cpu().numpy()

        self.log('val_accuracy', self.accuracy.compute(), prog_bar=True)
        self.log('val_precision', self.precision.compute(), prog_bar=True)
        self.log('val_recall', self.recall.compute(), prog_bar=True)
        self.log('val_f1', self.f1.compute(), prog_bar=True)

        self.plot_confusion_matrix(cm)

        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.confusion_matrix.reset()

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.float())
        predictions = (torch.sigmoid(y_hat) > self.threshold).int()
        y = y.int()
        accuracy = self.accuracy(predictions, y)

        self.log('test_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('test_accuracy', accuracy, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  mode='max',
                                                                  factor=self.scheduler_factor,
                                                                  patience=self.scheduler_patience)
        return (
            {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'monitor': 'val_f1',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        )

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Authentic', 'Fake'],
                    yticklabels=['Authentic', 'Fake'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f"Epoch: {str(self.current_epoch+1)}")
        plt.tight_layout()
        plt.show()
