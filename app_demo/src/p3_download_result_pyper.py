import streamlit as st
import os
import logging
import requests
import numpy as np
import pandas as pd
import pyper
import io
import json
import warnings
import zipfile
from datetime import datetime

warnings.filterwarnings("ignore")

styles = {'material-icons':{'color': 'red'},
          'text-icon-link-close-container': {'box-shadow': '#3896de 0px 4px'},
          'notification-text': {'':''},
          'close-button':{'':''},
          'link':{'':''}}

# Initialize R environment
r = pyper.R()
r("library(webchem)")
r("library(rJava)")
r("library(rcdklibs)")
r("library(rcdk)")
r("library(fingerprint)")

# Configure logging
logging.basicConfig(
    filename='runtime.log',  # Tên file log
    level=logging.INFO,         # Mức độ logging
    format='%(asctime)s - %(message)s',  # Định dạng log
    datefmt='%Y-%m-%d %H:%M:%S'  # Định dạng thời gian
)

def zip_files(file_list, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w') as zipf:
        for file in file_list:
            zipf.write(file, os.path.basename(file))
    return output_zip_path

# Function to log access information
def log_access(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"{current_time} - P3. Download   - {message}")
    print(f"{current_time} - P3. Download   - {message}")

def calculate_similarity(smiles1, smiles2):
    r.assign("smiles1", smiles1)
    r.assign("smiles2", smiles2)
    
    # Tạo fingerprint từ SMILES và tính toán độ tương đồng
    r("""
    mol1 <- parse.smiles(smiles1)[[1]]
    mol2 <- parse.smiles(smiles2)[[1]]
    fp1 <- get.fingerprint(mol1, type='circular')
    fp2 <- get.fingerprint(mol2, type='circular')
    similarity <- distance(fp1, fp2, method='tanimoto')
    """)
    
    similarity = r.get("similarity")
    return similarity

def download_result(file_names=None, sample_count=None):
    st.image("app_demo/image/3_download.png")
    if len(file_names) == 0:
        message = "Failed: No file uploaded for processing"
        st.error(message, icon="❌")
        log_access(message)
    count = 1
    list_output_path = []
    for file_name in file_names:
        if file_name is not None:
            try:
                df = pd.read_excel(file_name, sheet_name="Sheet1")
                experiment_df = pd.read_excel(file_name, sheet_name="Experiment value")
                if df.empty:
                    message = "Failed: The uploaded file is empty."
                    st.warning(message)
                    log_access(message)
                    return

                filtered_df = df[['Title', 'Area', 'SMILES']].dropna()
                # st.subheader("Filtered Data", divider='rainbow')
                # st.dataframe(filtered_df, height=350, width=1500)

                smiles_list = list(filtered_df["SMILES"])

                with open('config.json', 'r') as file:
                    config = json.load(file)

                IS_area = experiment_df['IS area'][0]
                mean = list(experiment_df['Mean']) 
                target_smiles = config.get('target_smiles')

                # Tạo ma trận tương đồng
                similarity_matrix = np.zeros((len(smiles_list), len(target_smiles)))
                for i, smiles1 in enumerate(smiles_list):
                    for j, smiles2 in enumerate(target_smiles):
                        similarity_matrix[i, j] = calculate_similarity(smiles1, smiles2)

                similarity_df = pd.DataFrame(similarity_matrix, columns=target_smiles, index=smiles_list)

                file_name_excel = file_name.name
                try:
                    samplei = file_name_excel.split('_')[0]
                except:
                    samplei = file_name.name.split(".xlsx")[0]

                sample = np.full((len(smiles_list),), samplei)
                semi_quan = pd.DataFrame({'Sample': sample})
                semi_quan['Title'] = filtered_df['Title'].values
                semi_quan['Best match'] = np.argmax(similarity_df.values, axis=1) + 1
                semi_quan['Area'] = filtered_df['Area'].values

                RF = []
                for i in range(len(semi_quan)):
                    rfi = semi_quan['Area'].values[i] / (IS_area * mean[semi_quan['Best match'].values[i] - 1]) * 40
                    RF.append(rfi)
                semi_quan['RF concerntration'] = RF

                data_info = pd.DataFrame()
                for i in range(len(smiles_list)):
                    num1 = [semi_quan['Title'].values[i], str('SMILES: ') + similarity_df.index[i],
                            semi_quan['Best match'].values[i], semi_quan['Area'].values[i],
                            semi_quan['RF concerntration'].values[i]]
                    num2 = np.full(5, np.nan, dtype=object)
                    array_str = np.array2string(similarity_df.iloc[i].values, separator=', ', max_line_width=np.inf)[1:-1]
                    num2[1] = str('Similarities: ') + array_str
                    num3 = np.full(5, np.nan, dtype=object)
                    max_similarityi = str(similarity_df.iloc[i].values.max())
                    best_matchi = target_smiles[np.argmax(similarity_df.iloc[i].values)]
                    num3[1] = str('Best Match: ') + best_matchi + str(' with Similarity: ') + max_similarityi
                    numpyi = np.stack((num1, num2, num3, np.full(5, np.nan, dtype=object)), axis=0)
                    dfi = pd.DataFrame(numpyi, columns=['Title', 'Information', 'Best match', 'Area', 'RF concentration'])
                    data_info = pd.concat([data_info, dfi], axis=0)

                # if count <= sample_count:
                #     st.subheader("Semi-quantification table", divider='rainbow')
                #     st.dataframe(semi_quan, height=500, width=1500)
                    # st.subheader("Sheet 2: Addition Output", divider='rainbow')
                    # st.dataframe(data_info, height=350, width=1500)

                output_path = f'app_demo/output/{file_name.name.split(".xlsx")[0]}_output.xlsx'
                list_output_path.append(output_path)
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    semi_quan.to_excel(writer, sheet_name='Sheet1', index=False)
                    data_info.to_excel(writer, sheet_name='Sheet2', index=False)     
                

            except Exception as e:
                message = f"Failed: Error during file processing. Error: {e}"
                st.error(message, icon="❌")
                log_access(message)
        
        count += 1
    if count > 1:
        st.subheader("Semi-quantification table", divider='rainbow')
        st.dataframe(semi_quan, height=500, width=1500)
        print(list_output_path)
        st.success("The result has been processed and is ready for download!", icon="✅")
        csv = convert_df(output_path)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name_output = f"app_demo/output/smile_{current_time}_output.zip"
        file_zip = zip_files(list_output_path, file_name_output)
        log_access(f"Success: Processed and exported results for file /{file_name_output}")
        st.sidebar.subheader("Download Result", divider='rainbow')

        with open(file_zip, "rb") as fp:
            btn = st.sidebar.download_button(
                label="Export to Excel",
                data=fp,
                file_name=f"smile_{current_time}_output.zip",
                mime="application/zip"
            )

        # st.sidebar.download_button(
        #     label="Export to Excel",
        #     data=csv,
        #     file_name=file_name_output
        # )

def convert_df(file_excel):
    dfs = pd.read_excel(file_excel, sheet_name=None)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    excel_data = output.getvalue()
    output.close()
    return excel_data
