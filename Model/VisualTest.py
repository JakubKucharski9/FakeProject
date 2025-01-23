from nike_pack import *
from Train import photo_transforms


def grad_cam_analysis(model, image, target_layer, target_class):
    model.eval()

    def forward_hook(module, input, output):
        model.feature_map = output

    def backward_hook(module, grad_in, grad_out):
        model.gradients = grad_out[0]

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    output = model(image)
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    gradients = model.gradients.cpu().detach()
    feature_map = model.feature_map.cpu().detach()

    weights = gradients.mean(dim=(2, 3))
    grad_cam_map = torch.zeros(feature_map.shape[2:])

    for i, w in enumerate(weights[0]):
        grad_cam_map += w * feature_map[0, i]

    grad_cam_map = torch.nn.functional.relu(grad_cam_map)
    grad_cam_map = grad_cam_map / grad_cam_map.max()

    handle_forward.remove()
    handle_backward.remove()

    return grad_cam_map.numpy()


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
    ax1.set_title("But testowy")
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

    fig.suptitle("Explainable AI Results", fontsize=16)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.show()


def predict_photo(model, image_path, device, transform, threshold=.5):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [
        "C:\\Users\\kuba\\PycharmProjects\\NikeProject\\Models\\model_9440_highthreshold.pth"
    ]
    test_dir = "C:\\Users\\kuba\\PycharmProjects\\NikeProject\\LC"

    for root, dirs, files in os.walk(test_dir):
        for image_file in files:
            image_path = os.path.join(root, image_file)
            image = Image.open(image_path).convert("RGB")
            transform = photo_transforms["test"]
            image_tensor = transform(image).unsqueeze(0).to(device)
            for model_path in models:
                model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
                model.to(device)

                target_layer = model.features[-1]

                grad_cam_map = grad_cam_analysis(model, image_tensor, target_layer, target_class=0)
                prediction, probability = predict_photo(model, image_path, device, transform, threshold=.5)

                plot_results(image_tensor, grad_cam_map, model_path, image_path, prediction=prediction, probability=probability)
