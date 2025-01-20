from __init__ import *


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

        return image, label


photo_transforms = {
    "train": transforms.Compose([
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

    ]),
    "test": transforms.Compose([
        transforms.ToImage(),

        transforms.Resize(600, interpolation=InterpolationMode.BICUBIC),

        transforms.CenterCrop(480),

        transforms.ToDtype(torch.float32, scale=True),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])}


def find_best_accuracy(best_accuracy_path):
    try:
        with open(best_accuracy_path, 'r') as file:
            best_accuracy = float(file.read())
    except FileNotFoundError:
        best_accuracy = -float('inf')

    return best_accuracy


def save_best_model(model, accuracy, best_model_path, best_accuracy_path):

    best_accuracy = find_best_accuracy(best_accuracy_path)

    if accuracy > best_accuracy:
        torch.save(model.state_dict(), best_model_path)
        with open(best_accuracy_path, 'w') as file:
            file.write(str(accuracy))
        tqdm.write(f"Accuracy została poprawiona. Best accuracy: {accuracy:.4f}")
    else:
        tqdm.write(f"Accuracy nie została poprawiona. Best accuracy: {best_accuracy:.4f}")


def calculate_optimal_threshold(labels, probabilities):
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]


def training(model, num_epochs, train_dataloader, test_dataloader, optimizer, criterion, scheduler, best_model_path, best_accuracy_path, device):
    torch.cuda.empty_cache()

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.squeeze(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            torch.cuda.empty_cache()

        tqdm.write(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_dataloader)}")

        model.eval()

        all_labels, all_probabilities = [], []

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                probabilities = torch.sigmoid(outputs).squeeze()
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Optimal threshold
        optimal_threshold = calculate_optimal_threshold(all_labels, all_probabilities)
        tqdm.write(f"Optimal threshold for epoch {epoch + 1}: {optimal_threshold:.4f}")
        all_predictions = (np.array(all_probabilities) >= optimal_threshold).astype(float)

        # Metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        tqdm.write(
            f"Epoch {epoch + 1} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Update the learning rate
        scheduler.step(accuracy)

        if (epoch + 1) % 5 == 0 and epoch != 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")

        save_best_model(model=model, accuracy=accuracy, best_model_path=best_model_path,
                        best_accuracy_path=best_accuracy_path)


if __name__ == "__main__":
    load_dotenv()
    dataset_unprocessed = load_dataset(os.getenv("DATASET_UNPROCESSED"))
    dataset_autoprocessed = load_dataset(os.getenv("DATASET_AUTOPROCESSED"))
    dataset_manualprocessed = load_dataset(os.getenv("DATASET_MANUALPROCESSED"))

    current_dataset = dataset_autoprocessed

    batch_size = 16

    dataset_train_loader_to_pytorch = ToPytorchDataset(current_dataset["train"], transform=photo_transforms["train"])
    train_dataloader = DataLoader(dataset_train_loader_to_pytorch, batch_size=batch_size, shuffle=True)

    dataset_test_loader_to_pytorch = ToPytorchDataset(dataset_unprocessed["test"], transform=photo_transforms["test"])
    test_dataloader = DataLoader(dataset_test_loader_to_pytorch, batch_size=batch_size, shuffle=False)

    weights = EfficientNet_V2_M_Weights.DEFAULT
    model = efficientnet_v2_m(weights=weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Used for Finetuning
    for param in model.features.parameters():
        param.requires_grad = False

    for param in list(model.features.children())[-5].parameters():
        param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(model.classifier[1].in_features, 1)
    )
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.7)

    criterion = torch.nn.BCEWithLogitsLoss()

    best_model_path = os.getenv("BEST_MODEL_PATH")
    best_accuracy_path = os.getenv("BEST_ACCURACY_PATH")

    num_epochs = 30


    # Fine-tuning
    model_path = "model_9000_unprocessed.pth"
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

    # Train the model
    training(model=model, num_epochs=num_epochs, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                optimizer=optimizer, criterion=criterion, scheduler=scheduler, best_model_path=best_model_path,
                best_accuracy_path=best_accuracy_path, device=device)
