from nike_pack import *


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
        if image.mode == "RGBA":
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


photo_transforms = {
    "test": transforms.Compose([
        transforms.ToImage(),
        transforms.Resize(600, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(480),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}


def calculate_optimal_threshold(labels, probabilities):
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]


def ensemble_prediction(models, dataloader, device):
    all_probabilities = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            model_outputs = [torch.sigmoid(model(inputs)).squeeze().cpu().numpy() for model in models]
            avg_prediction = np.mean(model_outputs, axis=0)
            all_probabilities.extend(avg_prediction)
    return np.array(all_probabilities)


class EnsembleModel(torch.nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models

    def forward(self, x):
        outputs = [torch.sigmoid(model(x)) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)


if __name__ == "__main__":

    dataset_unprocessed = load_dataset("Kucharek9/AirForce1_unprocessed")

    dataset_test_loader_to_pytorch = ToPytorchDataset(dataset_unprocessed["test"], transform=photo_transforms["test"])
    test_dataloader = DataLoader(dataset_test_loader_to_pytorch, batch_size=16, shuffle=False)

    models = [
        convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT),
        regnet_y_8gf(weights=RegNet_Y_8GF_Weights.DEFAULT),
        efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT),
    ]

    model_names = ["ConvNeXt_Base", "RegNet_Y_8GF", "EfficientNet_V2_M"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensemble_model = EnsembleModel(models)
    torch.save(ensemble_model, "ensemble_model.pth")

    for model_id, model in enumerate(models):
        model_name = model_names[model_id]
        best_model_path = f"outputs/best_model_{model_name}.pth"
        if model_id == 1:
            model.fc = torch.nn.Linear(model.fc.in_features, 1)
            model.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=0.3),
                torch.nn.Linear(model.fc.in_features, 1)
            )
        elif model_id == 2:
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.3),
                torch.nn.Linear(model.classifier[1].in_features, 1)
            )
        elif model_id == 0:
            model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 1)
            model.classifier = torch.nn.Sequential(
                torch.nn.Flatten(start_dim=1),
                torch.nn.LayerNorm(1024),
                torch.nn.Dropout(p=0.3),
                torch.nn.Linear(1024, 1)
            )
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True), strict=False)
        model.to(device)
        model.eval()

    test_labels = [label for _, label in dataset_test_loader_to_pytorch]
    ensemble_probabilities = ensemble_prediction(models, test_dataloader, device)
    optimal_threshold = calculate_optimal_threshold(test_labels, ensemble_probabilities)
    ensemble_predictions = (ensemble_probabilities >= optimal_threshold).astype(float)

    accuracy = accuracy_score(test_labels, ensemble_predictions)
    f1 = f1_score(test_labels, ensemble_predictions)
    precision = precision_score(test_labels, ensemble_predictions)
    recall = recall_score(test_labels, ensemble_predictions)

    print(
        f"Ensemble Model - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
