from nike_pack import *


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
    return heatmap, predicted_class



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoints = [
        "checkpoints/model_epoch=30_val_accuracy=0.9906.ckpt"
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
                    model = LightningModel.load_from_checkpoint(
                        checkpoint_path,
                        num_classes=1,
                        learning_rate=1e-4,
                        weight_decay=1e-4
                    )
                    model.to(device)
                    model.eval()

                    vis, y_hat = attribute(image, model)

                    plt.figure(figsize=(10, 10))
                    plt.title(f"Prediction: {y_hat}", fontsize=20)
                    plt.imshow(vis)
                    plt.axis('off')
                    plt.show()
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
