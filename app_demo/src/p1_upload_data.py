import streamlit as st
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

def upload_data(file_name=None):
    st.image("app_demo/image/1_upload_data.png")
    #st.subheader("📂 Please upload the Excel file you want to process.", divider='rainbow')
    # uploaded_file = st.file_uploader("", type=["xlsx"], accept_multiple_files=False)

    if file_name is not None: 
        df = pd.read_excel(file_name) 
        # Check if the DataFrame is empty
        if df.empty:
            st.warning("The uploaded file is empty. Please upload a file with data.")
            return
        st.subheader("Original Data", divider='rainbow')
        st.write(df) 

