from nike_pack import *


class LightningModel(LightningModule):
    def __init__(self, num_classes, learning_rate, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.model.classifier[1].in_features, num_classes)
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")

    def forward(self, x):
        return self.model(x).squeeze(dim=1)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.float()
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        predictions = torch.sigmoid(y_hat) > 0.5
        accuracy = self.accuracy(predictions, y.int())

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_accuracy', accuracy, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = y.float()
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        predictions = torch.sigmoid(y_hat) > 0.5

        self.log('val_accuracy', self.accuracy(predictions, y.int()), on_epoch=True, on_step=False)
        self.log('val_precision', self.precision(predictions, y.int()), on_epoch=True, on_step=False)
        self.log('val_recall', self.recall(predictions, y.int()), on_epoch=True, on_step=False)
        self.log('val_f1', self.f1(predictions, y.int()), on_epoch=True, on_step=False)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
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

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y = y.float()
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        predictions = torch.sigmoid(y_hat) > 0.5
        accuracy = self.accuracy(predictions, y.int())

        self.log('test_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('test_accuracy', accuracy, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.7, patience=5)
        return (
            {
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
                'monitor': 'val_f1'
            }
        )
