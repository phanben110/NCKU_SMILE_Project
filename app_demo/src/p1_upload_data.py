# import streamlit as st
# import pandas as pd
# import os
# import warnings
# warnings.filterwarnings("ignore")

# def upload_data(file_name=None):
#     st.image("app_demo/image/1_upload_data.png")
#     #st.subheader("📂 Please upload the Excel file you want to process.", divider='rainbow')
#     # uploaded_file = st.file_uploader("", type=["xlsx"], accept_multiple_files=False)

#     if file_name is not None: 
#         df = pd.read_excel(file_name) 
#         st.success("Upload data successful", icon="✅")
#         # Check if the DataFrame is empty
#         if df.empty:
#             st.warning("The uploaded file is empty. Please upload a file with data.")
#             return
#         st.subheader("Original Data", divider='rainbow')
#         st.write(df) 
        

#     elif file_name is None:
#         st.error("Please upload a data file to start processing.", icon="❌")


import streamlit as st
import pandas as pd
import json
import os
import warnings

warnings.filterwarnings("ignore")

# Function to upload and display Excel data
def upload_data(file_name=None):
    st.image("app_demo/image/1_upload_data.png")
    # st.subheader("📂 Please upload the Excel file you want to process.", divider='rainbow')
    # uploaded_file = st.file_uploader("", type=["xlsx"], accept_multiple_files=False)

    if file_name is not None: 
        df = pd.read_excel(file_name) 
        st.success("Upload data successful", icon="✅")
        # Check if the DataFrame is empty
        if df.empty:
            st.warning("The uploaded file is empty. Please upload a file with data.")
            return
        st.subheader("Original Data", divider='rainbow')
        st.write(df) 
    elif file_name is None:
        st.error("Please upload a data file to start processing.", icon="❌")

# Function to read and display JSON configuration
def manage_config(file_path='config.json'):
    # Check if the JSON file exists
    if not os.path.exists(file_path):
        st.error(f"The file {file_path} does not exist.", icon="❌")
        return

    # Read the JSON file
    with open(file_path, 'r') as file:
        config = json.load(file)

    # # Display JSON configuration

    # Editing form
    with st.form("config_edit_form"):
        st.subheader("Edit Configuration", divider='rainbow')
        is_area = st.number_input("IS_area", value=config.get('IS_area', 0))
        mean = st.text_input("Mean (comma-separated)", value=", ".join(map(str, config.get('mean', []))))
        target_smiles = st.text_area("Target SMILES (one per line)", value="\n".join(config.get('target_smiles', [])))
        submit_button = st.form_submit_button("Save Changes")

        if submit_button:
            try:
                # Update the configuration dictionary
                updated_config = {
                    'IS_area': is_area,
                    'mean': [float(x) for x in mean.split(',')],
                    'target_smiles': [x.strip() for x in target_smiles.split('\n')]
                }

                # Save the updated configuration to the file
                with open(file_path, 'w') as file:
                    json.dump(updated_config, file, indent=4) 

                with open(file_path, 'r') as file:
                    config = json.load(file)

                st.success("Configuration updated successfully!", icon="✅")
                st.subheader("Configuration", divider='rainbow')
                st.json(config)

            except ValueError as e:
                st.error(f"Error updating configuration: {e}", icon="❌")
            

    
