import streamlit as st
import pandas as pd
import json
import os
import logging
import warnings
from datetime import datetime

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

# Function to upload and display Excel data
def upload_data(file_names=None, sample_count=None):
    st.image("app_demo/image/1_upload_data.png")
    
    if len(file_names) == 0:
        message = "Failed: No file uploaded."
        st.error(message, icon="❌")
        log_access(message)
    count = 1
    for file_name in file_names:
        if file_name is not None:
            try:
                df = pd.read_excel(file_name)
                if df.empty:
                    message = "Failed: Uploaded file is empty."
                    st.warning(message)
                    log_access(message)
                else:
                    if count <= sample_count:
                        if count == 1: 
                            st.success("Upload data successful", icon="✅")
                        st.subheader(f"{count}. File Name: {file_name.name}", divider='rainbow')
                        st.write(df)
                    message = f"Success: Uploaded {file_name.name} data successfully."
                    log_access(message)
            except Exception as e:
                message = f"Failed: Error processing the uploaded file. Error: {e}"
                st.error(message, icon="❌")
                log_access(message)

        count += 1
# Function to read and display JSON configuration
def manage_config(file_path='config.json'):
    log_access("Accessed Manage Config Page")

    if not os.path.exists(file_path):
        message = f"Failed: The file {file_path} does not exist."
        st.error(message, icon="❌")
        log_access(message)
        return

    with open(file_path, 'r') as file:
        config = json.load(file)

    with st.form("config_edit_form"):
        st.subheader("Edit Configuration", divider='rainbow')
        # is_area = st.number_input("IS_area", value=config.get('IS_area', 0))
        # mean = st.text_input("Mean (comma-separated)", value=", ".join(map(str, config.get('mean', []))))
        target_smiles = st.text_area("Target SMILES (one per line)", value="\n".join(config.get('target_smiles', [])))
        submit_button = st.form_submit_button("Save Changes")

        if submit_button:
            try:
                updated_config = {
                    'target_smiles': [x.strip() for x in target_smiles.split('\n')]
                }

                with open(file_path, 'w') as file:
                    json.dump(updated_config, file, indent=4)

                with open(file_path, 'r') as file:
                    config = json.load(file)

                message = "Configuration updated successfully!"
                st.success(message, icon="✅")
                log_access(message)
                st.subheader("Configuration", divider='rainbow')
                st.json(config)

            except ValueError as e:
                message = f"Failed: Error updating configuration. Error: {e}"
                st.error(message, icon="❌")
                log_access(message)
