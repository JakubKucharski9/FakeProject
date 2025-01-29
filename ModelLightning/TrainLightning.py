from nike_pack import *


if __name__ == "__main__":
    load_dotenv()
    dataset_unprocessed = load_dataset(os.getenv("DATASET_UNPROCESSED"))
    dataset_autoprocessed = load_dataset(os.getenv("DATASET_AUTOPROCESSED"))
    dataset_manualprocessed = load_dataset(os.getenv("DATASET_MANUALPROCESSED"))
    dataset_clean = load_dataset("Kucharek9/AF1collection")

    batch_size = 16
    num_workers = min(4, cpu_count())

    photo_transforms = photo_transforms()

    train_dataloader = ToPytorchDataset.dataloader_from_hf(dataset=dataset_clean["train"],
                                                           transform=photo_transforms["train"],
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           num_workers=num_workers,
                                                           persistent_workers=True)

    val_dataloader = ToPytorchDataset.dataloader_from_hf(dataset=dataset_clean["validation"],
                                                         transform=photo_transforms["test"],
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         num_workers=num_workers,
                                                         persistent_workers=True)

    test_dataloader = ToPytorchDataset.dataloader_from_hf(dataset=dataset_clean["test"],
                                                          transform=photo_transforms["test"],
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          num_workers=num_workers,
                                                          persistent_workers=True)

    num_classes = 1
    learning_rate = 1e-3
    weight_decay = 1e-4
    model_to_use = "efficientnetv2" # efficientnetv2/ regnet/ efficientnetb4/ efficientnetb7

    model = LightningModel(num_classes=num_classes,
                           learning_rate=learning_rate,
                           weight_decay=weight_decay,
                           threshold=0.5,
                           scheduler_factor=0.5,
                           scheduler_patience=5,
                           model_to_use=model_to_use)

    logger = TensorBoardLogger(f"logs/tests_{model_to_use}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoints = ModelCheckpoint(
        monitor="val_accuracy",
        mode="max",
        save_top_k=3,
        dirpath="checkpoints/",
        filename="model_{epoch}_{val_accuracy:.4f}",
    )

    trainer = Trainer(
        callbacks=[checkpoints],
        max_epochs=100,
        accelerator=device,
        devices=1,
        logger=logger,
        log_every_n_steps=0,
        accumulate_grad_batches=1
    )

    torch.set_float32_matmul_precision('high')

    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)
