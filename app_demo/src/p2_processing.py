import streamlit as st
import os
import requests
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def processing(file_name=None): 
    st.image("app_demo/image/2_processing.png")
    # Read the Excel file into a DataFrame
    if file_name is not None:

        df = pd.read_excel(file_name)
        # Check if the DataFrame is empty
        if df.empty:
            st.warning("The uploaded file is empty. Please upload a file with data.")
            return

        filtered_df = df[['Title', 'Area', 'SMILES']] 
        filtered_df.head()
        filtered_df = filtered_df.dropna() 
        st.subheader(" Filtered Data", divider='rainbow')
        st.dataframe(filtered_df, height=500, width=1500)


    elif file_name is None:
        st.error("Please upload a data file to start processing.", icon="❌")

