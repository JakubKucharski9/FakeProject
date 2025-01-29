from nike_pack import *
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable


def to_gray_image(x):
    x = x - x.min()
    x = x / x.max()
    x *= 255
    return np.array(x, dtype=np.uint8)


def overlay_heatmap(img, grad):
    img_np = np.array(img)
    grad = grad.squeeze().detach().cpu().numpy()
    grad_img = to_gray_image(grad)

    if len(grad_img.shape) == 2:
        grad_img = np.expand_dims(grad_img, axis=-1)

    heatmap = cv2.applyColorMap(grad_img, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
    return overlay


def attribute(image, model):
    transform = photo_transforms["test"]
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probability = torch.sigmoid(logits).item()
        predicted_class = 1 if probability > 0.5 else 0

    layer_grad_cam = LayerGradCam(model, model.model.features)
    attr = layer_grad_cam.attribute(image_tensor, target=None)

    unsampled_attr = torch.nn.functional.interpolate(
        attr, size=(image.height, image.width), mode='bicubic', align_corners=False
    )

    heatmap = overlay_heatmap(image, unsampled_attr)
    return heatmap, predicted_class, probability * 100, image_tensor


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
    ax1.set_title(image_path.lstrip("../Test_photos").rstrip(".jpg"))
    ax1.axis("off")

    ax2 = axes[1]

    im = ax2.imshow(grad_cam_map, cmap="jet")
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

    fig.suptitle(model_path.lstrip("checkpoints/"), fontsize=16)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoints = [
        "checkpoints/best_model_epoch=7_val_accuracy=0.9769.ckpt"
    ]

    photo_transforms = photo_transforms()
    test_dir = "../Test_photos"

    for root, _, files in os.walk(test_dir):
        for image_file in files:
            if not image_file.lower().endswith(('png', 'jpg', 'jpeg')):
                continue

            image_path = os.path.join(root, image_file)
            try:
                image = Image.open(image_path).convert("RGB")
                for checkpoint_path in checkpoints:
                    model = LightningModel(num_classes=1, learning_rate=1e-4, weight_decay=1e-4)
                    checkpoint = torch.load(checkpoint_path, map_location=device)

                    # Extract only state_dict if checkpoint includes additional metadata
                    model.load_state_dict(checkpoint['state_dict'])
                    model.to(device)
                    model.eval()

                    vis, y_hat, prob, img_tensor = attribute(image, model)

                    plot_results(img_tensor, vis, checkpoint_path, image_path, y_hat, prob)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
