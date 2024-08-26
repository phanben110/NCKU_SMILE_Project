import streamlit as st
from streamlit_option_menu import option_menu
import re
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from app_demo.src.p1_upload_data import *
from app_demo.src.p2_processing import *
# from app_demo.src.p3_download_result_rpy2 import *
from app_demo.src.p3_download_result_pyper import *
from app_demo.src.footer import settingFooter
import warnings
import logging
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    filename='runtime.log',  # Tên file log
    level=logging.INFO,     # Mức độ logging
    format='%(asctime)s - %(message)s',  # Định dạng log
    datefmt='%Y-%m-%d %H:%M:%S'  # Định dạng thời gian
)

# Function to log access information
def log_access(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"{current_time} - P1. Upload     - {message}")
    print(f"{current_time} - P1. Upload     - {message}")

# Create an option menu for the main menu in the sidebar
st.set_page_config(page_title="Semi-quantitative", page_icon="app_demo/image/Icon_chemitry.png", layout="wide")
#st.set_page_config(page_title="Semi-quantitative", page_icon="app_demo/image/Icon_chemitry.png", layout="wide",  theme={"primaryColor": "#4CAF50"})
st.sidebar.image("app_demo/image/logo_NCKU.jpeg", use_column_width=True)

with st.sidebar:
    selected = option_menu("Main Menu", ["1. Upload Data", "2. Processing", "3. Download Result"],
                           icons=["cloud-upload-fill", "cpu-fill", "cloud-arrow-down-fill" ], menu_icon="bars", default_index=0)
# Based on the selected option, you can display different content in your web application
# page for select icon https://icons.getbootstrap.com/

sample_count = st.sidebar.number_input("Modify it to display the number of samples", min_value = 1, value=3) 
log_access(f"Number of samples displayed: {sample_count}")

#st.sidebar.subheader("Please upload the Excel files", divider='rainbow')
uploaded_files = st.sidebar.file_uploader(label="Please upload the Excel files", type=["xlsx"], accept_multiple_files=True) 

# settingFooter()

if selected == "1. Upload Data":
    upload_data(uploaded_files, sample_count)   
    if len(uploaded_files) > 0:
        manage_config()  
elif selected == "2. Processing":
    processing(uploaded_files, sample_count)
elif selected == "3. Download Result":
    download_result(uploaded_files, sample_count)

