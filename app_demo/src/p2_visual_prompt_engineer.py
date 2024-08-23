import streamlit as st
from os.path import expanduser, join
import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from evaluation_utils import img_preprocess, denorm
import warnings
warnings.filterwarnings("ignore")

# Function to load sample images
def load_sample(input_image, input_mask):
    tf = transforms.Compose([
        transforms.ToTensor()
    ])
    tf2 = transforms.Compose([
        transforms.ToTensor()
    ])

    # cropped_image = crop_center_square(input_image)
    # resized_image = cropped_image.resize((224, 224))

    # cropped_mask = crop_center_square(input_mask)
    # resized_mask = cropped_mask.resize((224, 224))

    inp1 = [None, tf(input_image), tf2(input_mask)]
    inp1[1] = inp1[1].unsqueeze(0)
    inp1[2] = inp1[2][:1]
    
    return inp1

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

# Function to preprocess images with different options
def all_preprocessing(inp1):
    return [
        img_preprocess(inp1),
        img_preprocess(inp1, colorize=True),
        # img_preprocess(inp1, outline=True ),        
        img_preprocess(inp1, blur=3),
        img_preprocess(inp1, bg_fac=0.1),
        # img_preprocess(inp1, bg_fac=0.5),
        img_preprocess(inp1, blur=3, bg_fac=0.5),        
        img_preprocess(inp1, blur=3, bg_fac=0.5, center_context=0.1),
    ]



from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

def visual_prompt_engineer():
    st.image("app_demo/image/2_Visual Prompt Engineer.png")

    uploaded_image = st.sidebar.file_uploader("Please upload the image", type=["png", "jpg"], accept_multiple_files=False) 
    if uploaded_image:
        st.sidebar.image(uploaded_image)
    uploaded_image_mask = st.sidebar.file_uploader("Please upload the image you want to use for segmentation!", type=["png", "jpg"], accept_multiple_files=False) 
    if uploaded_image_mask:
        st.sidebar.image(uploaded_image_mask)
    if uploaded_image and uploaded_image_mask: 

        input_image = Image.open(uploaded_image).convert("RGB")
        cropped_image = crop_center_square(input_image)
        resized_image = cropped_image.resize((352, 352)) 
        input_image_mask = Image.open(uploaded_image_mask).convert("RGB")
        cropped_image_mask = crop_center_square(input_image_mask)
        resized_image_mask = cropped_image_mask.resize((352, 352)) 
        col1, col2 = st.columns(2) 
        with col1:
            if uploaded_image:
                st.image(resized_image, caption="Support image")
        with col2:
            if uploaded_image_mask:
                st.image(resized_image_mask, caption="Support mask")

        # Load sample images using resized_image and resized_image_mask
        images_queries = load_sample(resized_image, resized_image_mask)
        
        # Preprocess images with selected options
        joint_image = all_preprocessing(images_queries)
        
        # Display processed images in two rows of three columns each
        num_images = len(joint_image)
        num_cols = 3
        num_rows = num_images // num_cols
        
        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col in range(num_cols):
                index = row * num_cols + col
                if index < num_images:
                    # Convert torch.Tensor to PIL.Image
                    img = joint_image[index]
                    pil_img = TF.to_pil_image(make_grid(img.cpu()))  # Assuming img is torch.Tensor
                    cols[col].image(pil_img, caption=f"Processed Image {index+1}")

        # st.pyplot()  # Ensure to call this only once at the end to display any plots
