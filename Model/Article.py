import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.transforms import transforms
from captum.attr import IntegratedGradients
from PIL import Image


def integrated_gradients_analysis(model, image, target_class):
    model.eval()

    # Inicjalizacja Integrated Gradients
    ig = IntegratedGradients(model)

    # Obliczanie atrybucji za pomocą Integrated Gradients
    attributions, _ = ig.attribute(
        inputs=image,
        target=target_class,
        return_convergence_delta=True
    )

    # Przetwarzanie atrybucji
    attributions = attributions.squeeze().cpu().detach().numpy()
    attributions = np.sum(attributions, axis=0)  # Suma przez kanały RGB

    return attributions


def visualize_integrated_gradients(image_tensor, attributions, title="Integrated Gradients"):
    # Odtworzenie obrazu wejściowego
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    for i in range(3):  # Odwrócenie normalizacji
        image_np[..., i] = image_np[..., i] * std[i] + mean[i]
    image_np = np.clip(image_np, 0, 1)

    # Skalowanie atrybucji
    attributions_normalized = attributions - attributions.min()
    attributions_normalized = attributions_normalized / attributions_normalized.max()

    # Wizualizacja
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(image_np)
    ax[0].axis("off")
    ax[0].set_title("Original Image")

    ax[1].imshow(image_np)
    ax[1].imshow(attributions_normalized, cmap="jet", alpha=0.5)
    ax[1].axis("off")
    ax[1].set_title(title)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. Ładowanie modelu
    model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)  # Dostosowanie do binarnej klasyfikacji
    model.load_state_dict(torch.load("../Models/model_9440_highthreshold.pth", map_location="cpu"))  # Podaj ścieżkę do wytrenowanego modelu
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Przetwarzanie obrazu
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    Tests = [
        "../fake.jpg",
        "../legit.jpg",
    ]
    for photo in Tests:
        image_path = photo
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

        target_class = 0
        attributions = integrated_gradients_analysis(model, image_tensor, target_class)

        # 4. Wizualizacja
        visualize_integrated_gradients(image_tensor, attributions, title="Integrated Gradients for 'Fake'")
