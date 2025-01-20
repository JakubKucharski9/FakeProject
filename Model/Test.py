from nike_pack import *

def predict_photo(model, image_path, device, transform, threshold=.1):

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probability = torch.sigmoid(outputs).item()
        if probability >= threshold:
            prediction = 1
        else:
            prediction = 0

    return prediction, probability


if __name__ == "__main__":
    image_path = "../but5.jpg"

    transform = photo_transforms["test"]
    weights = EfficientNet_V2_M_Weights.DEFAULT
    model = efficientnet_v2_m(weights=weights)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)

    model_path = "model_9440_highthreshold.pth"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

    prediction, probability = predict_photo(model, image_path, device, transform, threshold=.19)

    print(f"Prediction: {prediction}, Probability: {probability:.4f}")