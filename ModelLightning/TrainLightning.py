from nike_pack import *


if __name__ == "__main__":
    load_dotenv()
    dataset_unprocessed = load_dataset(os.getenv("DATASET_UNPROCESSED"))
    dataset_autoprocessed = load_dataset(os.getenv("DATASET_AUTOPROCESSED"))
    dataset_manualprocessed = load_dataset(os.getenv("DATASET_MANUALPROCESSED"))

    batch_size = 16
    num_workers = min(4, cpu_count())

    photo_transforms = photo_transforms()

    train_dataloader = ToPytorchDataset.dataloader_from_hf(dataset=dataset_manualprocessed["train"],
                                                           transform=photo_transforms["train"],
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           num_workers=num_workers,
                                                           persistent_workers=True)

    val_dataloader = ToPytorchDataset.dataloader_from_hf(dataset=dataset_manualprocessed["test"],
                                                         transform=photo_transforms["test"],
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         num_workers=num_workers,
                                                         persistent_workers=True)

    test_dataloader = ToPytorchDataset.dataloader_from_hf(dataset=dataset_unprocessed["test"],
                                                          transform=photo_transforms["test"],
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          num_workers=num_workers,
                                                          persistent_workers=True)

    num_classes = 2
    learning_rate = 1e-4
    logger = TensorBoardLogger("logs/")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_pth = None
    model_pth = "../Models/model_9440.pth"

    if checkpoint_pth is not None:
        model = LightningModel.load_from_checkpoint(checkpoint_pth, num_classes=num_classes, learning_rate=learning_rate, freeze_bn=True)
    elif model_pth is not None:
        model = LightningModel(num_classes=num_classes, learning_rate=learning_rate)
        state_dict = torch.load(model_pth, map_location=torch.device(device))
        model.load_state_dict(state_dict, strict=False)
    else:
        model = LightningModel(num_classes=num_classes, learning_rate=learning_rate)

    checkpoints = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        dirpath="checkpoints/",
        filename="model_{epoch}_{val_accuracy:.4f}",
    )

    trainer = Trainer(
        callbacks=[checkpoints],
        max_epochs=50,
        accelerator=device,
        devices=1,
        logger=logger,
        log_every_n_steps=5,
        accumulate_grad_batches=1
    )

    torch.set_float32_matmul_precision('high')

    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)
