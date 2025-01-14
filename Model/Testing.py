from PIL import Image
import torch
import numpy as np
import cv2
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from Model.Train import photo_transforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from captum.attr import LayerGradCam

def preprocess_image(image, target_size):
    """Przygotowuje obraz do analizy przez model."""
    image = cv2.resize(image, target_size)
    image = np.transpose(image, (2, 0, 1))  # Zamiana na format CHW
    image = np.expand_dims(image, axis=0)  # Dodanie wymiaru batch
    image = torch.tensor(image, dtype=torch.float32) / 255.0  # Normalizacja
    return image

def compute_smooth_grad_cam(model, img_tensor, class_idx, target_layer, num_samples=25, std_dev=0.2):
    """Oblicza Smooth Grad-CAM przy użyciu Captum dla wybranego obrazu i klasy."""
    def forward_hook(module, input, output):
        model.feature_map = output

    def backward_hook(module, grad_in, grad_out):
        model.gradients = grad_out[0]

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    grad_cam_maps = []
    for _ in range(num_samples):
        noise = torch.normal(mean=0, std=std_dev, size=img_tensor.shape).to(img_tensor.device)
        noisy_image = img_tensor + noise
        noisy_image = noisy_image.requires_grad_(True)

        output = model(noisy_image)
        model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)

        gradients = model.gradients.cpu().detach()
        feature_map = model.feature_map.cpu().detach()

        weights = gradients.mean(dim=(2, 3))
        grad_cam_map = torch.zeros(feature_map.shape[2:])

        for i, w in enumerate(weights[0]):
            grad_cam_map += w * feature_map[0, i]

        grad_cam_map = torch.nn.functional.relu(grad_cam_map)
        grad_cam_maps.append(grad_cam_map.numpy())

    smoothed_map = np.mean(grad_cam_maps, axis=0)
    smoothed_map = smoothed_map / smoothed_map.max()

    handle_forward.remove()
    handle_backward.remove()

    return smoothed_map

def overlay_heatmap(heatmap, image, alpha=0.4):
    """Nakłada mapę ciepła na oryginalny obraz."""
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed_image = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
    return overlayed_image

def plot_results(original_image, grad_cam_map, model_path, image_path, prediction, probability):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image = original_image.squeeze(0).clone()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)

    image_np = image.permute(1, 2, 0).cpu().numpy().clip(0, 1)

    desired_size = (300, 300)
    image_np = cv2.resize(image_np, desired_size, interpolation=cv2.INTER_LINEAR)
    grad_cam_map_resized = cv2.resize(grad_cam_map, desired_size, interpolation=cv2.INTER_LINEAR)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1.05]})

    ax1 = axes[0]
    ax1.imshow(image_np)
    ax1.set_title(image_path.lstrip("../").rstrip("."))
    ax1.axis("off")

    ax2 = axes[1]
    ax2.imshow(image_np)
    im = ax2.imshow(grad_cam_map_resized, cmap="jet", alpha=0.5)
    ax2.set_title("Grad-CAM")
    ax2.axis("off")

    ax2.text(
        0.5, -0.1,
        f"Prediction: {prediction}\nProbability: {probability:.2f}%",
        fontsize=12,
        color="black",
        ha="center",
        va="center",
        transform=ax2.transAxes,
    )

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label="Grad-CAM Heatmap")

    fig.suptitle(model_path.rstrip(".pth").lstrip("model_"), fontsize=16)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.show()

def predict_photo(model, image_path, device, transform, threshold):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probability = torch.sigmoid(outputs).item()
        if probability >= threshold:
            prediction = "Fake"
            prediction_probability = probability * 100

        else:
            prediction = "Legit"
            prediction_probability = 100 - (probability * 100)

    return prediction, prediction_probability

if __name__ == "__main__":
    model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)

    models = [
        "model_9440_highthreshold.pth"
    ]

    Fakes = [
        "../but1.jpg",
        "../but5.jpg",
    ]

    Legit = [
        "../but6.jpg",
        "../but7.jpg",
        "../but8.jpg",
        "../but9.jpg",
    ]

    LegitCheck = [
        "../lc/1.jpg",
        "../lc/2.jpg",
        "../lc/3.jpg",
        "../lc/4.jpg",
    ]

    LegitCheck2 = [
        "../lc2/1.jpg",
        "../lc2/2.jpg",
        "../lc2/3.jpg",
        "../lc2/4.png",
        "../lc2/5.jpg",
    ]

    images = LegitCheck

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        transform = photo_transforms["test"]
        image_tensor = transform(image).unsqueeze(0).to(device)
        for model_path in models:
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
            model.to(device)
            target_layer = model.features[-1]
            grad_cam_map = compute_smooth_grad_cam(model, image_tensor, class_idx=0, target_layer=target_layer)
            prediction, probability = predict_photo(model, image_path, device, transform, threshold=.5)

            plot_results(image_tensor, grad_cam_map, model_path, image_path, prediction=prediction, probability=probability)
