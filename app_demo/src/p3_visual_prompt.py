import streamlit as st
import os
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPSegModel 
import streamlit as st
import os
import torch
import requests
from streamlit_chat import message
from streamlit.components.v1 import html
import cv2
from PIL import Image, ImageFilter
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print("using device:", device)

# ! wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip
# ! unzip -d weights -j weights.zip
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

def load_model(): 
    # load model
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval();
    # non-strict, because we only stored decoder weights (not CLIP weights)
    model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device(device)), strict=False);
    return model 

# Function to center-crop and resize the image
def crop_center_square(img):
    width, height = img.size
    new_edge = min(width, height)
    left = (width - new_edge) / 2
    top = (height - new_edge) / 2
    right = (width + new_edge) / 2
    bottom = (height + new_edge) / 2
    img_cropped = img.crop((left, top, right, bottom))
    return img_cropped

def image_feature(input_image): 
    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

    image = Image.open(input_image)

    inputs = processor(images=image, return_tensors="pt")

    image_features = model.get_image_features(**inputs)

    return image_features


def visual_prompt():
    st.image("app_demo/image/3_Visual Prompt.png")

    uploaded_image = st.sidebar.file_uploader("Please upload the image you want to use for segmentation!", type=["png", "jpg"], accept_multiple_files=False) 
    if uploaded_image:
        st.sidebar.image(uploaded_image)
    uploaded_image_visual = st.sidebar.file_uploader("Please upload the visual prompt", type=["png", "jpg"], accept_multiple_files=False)
    if uploaded_image_visual :  
        st.sidebar.image(uploaded_image_visual)
    
    if uploaded_image:
        input_image = Image.open(uploaded_image).convert("RGB")
        cropped_image = crop_center_square(input_image)
        resized_image = cropped_image.resize((352, 352))

        # Display the cropped and resized image
        with st.chat_message("user", avatar="app_demo/image/5_avatar_user.jpeg"):
            st.image(resized_image) 

    prompt = st.chat_input("Describe something that you want a model to segment?")
        
    if prompt and uploaded_image and uploaded_image_visual:
        with st.spinner('Model AI is analyzing...'):  # Add spinner while processing
            with st.chat_message("user", avatar="app_demo/image/5_avatar_user.jpeg"):
                st.write(prompt)
                st.image(uploaded_image_visual)
            
            model = load_model()
            visual_prompt = image_feature(uploaded_image_visual)
            print(visual_prompt.shape)
            
            transform = transforms.Compose([
                transforms.Resize((352, 352)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            img = transform(cropped_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                preds = model(img.repeat(4, 1, 1, 1), visual_prompt)[0]
            
            segmented_image = torch.sigmoid(preds[0][0]).cpu().numpy()

            # Convert segmented image to a PIL image, apply Gaussian blur, and back to numpy array
            segmented_image_resized = Image.fromarray((segmented_image * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5))
            segmented_image_resized = np.array(segmented_image_resized)

            # Create color_mask initialized to zeros
            color_mask = np.zeros((segmented_image_resized.shape[0], segmented_image_resized.shape[1], 3), dtype=np.uint8)

            # Threshold to identify significant pixels
            threshold_pixel = 100
            mask_indices = segmented_image_resized > threshold_pixel

            # Only keep the red channel, set others to 0
            color_mask[mask_indices, 0] = segmented_image_resized[mask_indices]

            # Convert resized original image to numpy array
            original_resized_np = np.array(resized_image)
            overlay = original_resized_np.copy()
            alpha = 0.5

            # Find connected components in the mask
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(color_mask[:, :, 0], connectivity=8)

            # Area threshold for filtering small regions
            area_threshold = 3600

            # Initialize new_mask to zeros
            new_mask = np.zeros_like(labels, dtype=np.uint8)

            # Loop through labels, skip label 0 (background)
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA] 
                print(f"label: {label}, area : {area}")
                if area > area_threshold:
                    # Keep only regions larger than area_threshold
                    new_mask[labels == label] = 255
                else : 
                    new_mask[labels == label] = 0
            # Create mask_indices from new_mask
            mask_indices = np.where(new_mask > 0)

            # Apply alpha blending only on remaining regions
            overlay[mask_indices[0], mask_indices[1], :] = (
                alpha * color_mask[mask_indices[0], mask_indices[1], :] + 
                (1 - alpha) * original_resized_np[mask_indices[0], mask_indices[1], :]
            )

            # Display the combined image
            with st.chat_message("assistant", avatar="app_demo/image/4_avatar_robot.png"):
                st.image(overlay)