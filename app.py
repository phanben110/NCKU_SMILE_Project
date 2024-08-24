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
from app_demo.src.p3_download_result import *
from app_demo.src.footer import settingFooter
import warnings
warnings.filterwarnings("ignore")



# Create an option menu for the main menu in the sidebar
st.set_page_config(page_title="Semi-quantitative", page_icon="app_demo/image/Icon_chemitry.png", layout="wide")
#st.set_page_config(page_title="Semi-quantitative", page_icon="app_demo/image/Icon_chemitry.png", layout="wide",  theme={"primaryColor": "#4CAF50"})
st.sidebar.image("app_demo/image/logo_NCKU.jpeg", use_column_width=True)

with st.sidebar:
    selected = option_menu("Main Menu", ["1. Upload Data", "2. Processing", "3. Download Result"],
                           icons=["cloud-upload-fill", "cpu-fill", "cloud-arrow-down-fill" ], menu_icon="bars", default_index=0)
# Based on the selected option, you can display different content in your web application
# page for select icon https://icons.getbootstrap.com/


st.sidebar.subheader("Please upload the Excel file", divider='rainbow')
uploaded_file = st.sidebar.file_uploader("", type=["xlsx"], accept_multiple_files=False) 

# settingFooter()

if selected == "1. Upload Data":
    upload_data(uploaded_file)   
    if uploaded_file is not None:
        manage_config()  
elif selected == "2. Processing":
    processing(uploaded_file)
elif selected == "3. Download Result":
    download_result(uploaded_file)

