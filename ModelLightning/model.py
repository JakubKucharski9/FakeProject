from nike_pack import *


class LightningModel(LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-4):
        super().__init__()
        self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.model.classifier[1].in_features, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()
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
        x_hat = self(x)
        loss = self.loss_fn(x_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
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
