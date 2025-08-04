import sys
import numpy as np
import matplotlib.cm as cm
import cv2
import streamlit as st
from PIL import Image
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_lightning.data_transforms import photo_transforms
from model_lightning.LightningVisualTest import grad_cam_analysis, predict_photo


def page():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "Model/best_model_EfficientNet_V2_M.pth"

    model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
    num_classes = 1
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.to(device)

    st.title("AF1 FAKE DETECTION")
    st.markdown("[Article](https://www.medium.com/@kuba_57640/how-artificial-intelligence-changes-game-in-fakes"
                "-detection-1360f82ace5c)", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload files to check authenticity", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
    result = 0
    if uploaded_files:
        for file in uploaded_files:
            image = Image.open(file).convert("RGB")
            transform = photo_transforms()["test"]
            image_tensor = transform(image).unsqueeze(0).to(device)
            prediction, probability = predict_photo(model, file, device, transform, threshold=0.3)
            if prediction == "Legit": result += 100 - probability
            elif prediction == "Fake": result += probability
            grad_cam_map = grad_cam_analysis(model, image_tensor, model.features[-1])

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
            image_denorm = image_tensor.squeeze(0).cpu() * std.cpu() + mean.cpu()
            image_np = image_denorm.permute(1, 2, 0).numpy().clip(0, 1)

            desired_size = (300, 300)
            image_np_resized = cv2.resize(image_np, desired_size, interpolation=cv2.INTER_LINEAR)
            grad_cam_map_resized = cv2.resize(grad_cam_map, desired_size, interpolation=cv2.INTER_LINEAR)

            heatmap = cm.jet(grad_cam_map_resized)[..., :3]
            overlay = (0.5 * image_np_resized + 0.5 * heatmap)
            overlay = (overlay * 255).astype(np.uint8)

            st.image(overlay)

        final_result = result/len(uploaded_files)
        if final_result > 30: st.write(f"Fake, confidence: {final_result:.2f}%")
        else: st.write(f"Legit, confidence: {100-final_result:.2f}%")

if __name__ == "__main__":
    page()