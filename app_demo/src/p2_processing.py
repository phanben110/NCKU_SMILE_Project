import streamlit as st
import os
import logging
import numpy as np
import pandas as pd
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    filename='runtime.log',  # Tên file log
    level=logging.INFO,         # Mức độ logging
    format='%(asctime)s - %(message)s',  # Định dạng log
    datefmt='%Y-%m-%d %H:%M:%S'  # Định dạng thời gian
)

# Function to log access information
def log_access(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"{current_time} - P2. Processing - {message}")
    print(f"{current_time} - P2. Processing - {message}")

# Function to process the uploaded Excel file
def processing(file_names=None, sample_count=None):
    st.image("app_demo/image/2_processing.png")

    if len(file_names) == 0:
        message = "Failed: No file uploaded for processing"
        st.error(message, icon="❌")
        log_access(message)

    count = 1
    for file_name in file_names:
    
        if file_name is not None:
            try:
                # Read the Excel file into a DataFrame
                df = pd.read_excel(file_name)
                
                if df.empty:
                    message = "Failed: The uploaded file is empty."
                    st.warning(message)
                    log_access(message)
                    return
                
                # Filter and display the data
                if count <= sample_count:
                    filtered_df = df[['Title', 'Area', 'SMILES']].dropna()
                    st.subheader(f"{count}. Data after filter of the file {file_name.name}", divider='rainbow')
                    st.dataframe(filtered_df, height=500, width=1500)
                    message = f"Success: {file_name.name} processed and filtered successfully."
                    log_access(message)
            
            except Exception as e:
                message = f"Failed: Error during data processing. Error: {e}"
                st.error(message, icon="❌")
                log_access(message)
        
        elif file_name is None:
            message = "Failed: No file uploaded for processing."
            st.error(message, icon="❌")
            log_access(message)
        count +=1 