
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from pothole import PotholeModel  

# Function to calculate perceived focal length

def calculate_perceived_focal_length(bbox):
    length = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]
    pixel_length = length  # Assuming length represents pixel length
    camera_distance = 90  # Fixed camera distance in centimeters
    return (pixel_length * camera_distance) / width

# Function to estimate dimensions
def estimate_dimensions(image, gt_bbox, model):
    with torch.no_grad():
        # Preprocess the image
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = F.to_tensor(image).unsqueeze(0)

        # Perform inference
        pred_bboxes = model(image)
        
        # Calculate perceived focal length for each detected pothole
        perceived_focal_lengths = []
        for pred_bbox in pred_bboxes:
            perceived_focal_length = calculate_perceived_focal_length(pred_bbox)
            perceived_focal_lengths.append(perceived_focal_length)

        # Calculate average perceived focal length
        average_perceived_focal_length = torch.mean(torch.tensor(perceived_focal_lengths)).item()

        return average_perceived_focal_length

# Load the model
model = PotholeModel()
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Streamlit app
st.title("Pothole Dimension Estimation")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Estimate Dimensions'):
        # Ground truth bounding box coordinates (xmin, ymin, xmax, ymax)
        gt_bbox = [0, 0, 100, 100]  # Replace with actual ground truth bbox

        # Estimate dimensions
        estimated_length = estimate_dimensions(image, gt_bbox, model)

        st.write(f"Estimated Average Perceived Focal Length: {estimated_length} cm")
