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

def calculate_similarity_df(filtered_df_i,target_smiles):
    smiles_list_i = list(filtered_df_i["SMILES"]) 
    # Tạo ma trận tương đồng
    similarity_matrix = np.zeros((len(smiles_list_i), len(target_smiles)))
    for i, smiles1 in enumerate(smiles_list_i):
        for j, smiles2 in enumerate(target_smiles):
            similarity_matrix[i, j] = calculate_similarity(smiles1, smiles2)
    # Chuyển kết quả thành định dạng DataFrame
    similarity_i = pd.DataFrame(similarity_matrix, columns=target_smiles, index=smiles_list_i)
    return similarity_i

def summary_1(similarity_i, filtered_df_i, file_name_i, IS_area, mean):

    try:
        sample_i = file_name_i.split('_')[0]
    except:
        sample_i = file_name_i.name.split(".xlsx")[0]
    sample_name_i = np.full((len(filtered_df_i),), sample_i)
    semi_quan_singe = pd.DataFrame({'Sample': sample_name_i})
    #Add Title column
    semi_quan_singe['Title'] = filtered_df_i['Title'].values
    #Add Best match column
    semi_quan_singe['Best match'] = np.argmax(similarity_i.values, axis=1) + 1
    #Add Area column
    semi_quan_singe['Area'] = filtered_df_i['Area'].values
    #Add RF concerntration column
    # mean = [11.5710449,2.314978868,1.980552596,0.892092811,1.280020852,3.950444301]
    RF = []
    for i in range(len(semi_quan_singe)):
        rfi = semi_quan_singe['Area'].values[i]/(IS_area*mean[semi_quan_singe['Best match'].values[i]-1])*40
        RF.append(rfi)
    semi_quan_singe['RF concerntration'] = RF
    return semi_quan_singe

def summary_2(semi_quan_i,similarity_df_i,target_smiles):
    ## Create Data_information
    data_info_i = pd.DataFrame()
    for i in range(similarity_df_i.values.shape[0]):
        num1 = [semi_quan_i['Title'].values[i], str('SMILES: ') + similarity_df_i.index[i], semi_quan_i['Best match'].values[i],semi_quan_i['Area'].values[i],semi_quan_i['RF concerntration'].values[i]]
        num2 = np.full(5, np.nan, dtype=object)
        array_str = np.array2string(similarity_df_i.iloc[i].values, separator=', ', max_line_width=np.inf)[1:-1]
        num2[1] = str('Similarities: ') + array_str
        num3 = np.full(5, np.nan, dtype=object)
        max_similarityi = str(similarity_df_i.iloc[i].values.max())
        best_matchi = target_smiles[np.argmax(similarity_df_i.iloc[i].values)]
        num3[1] = str('Best Match: ') + best_matchi + str(' with Similarity: ') + max_similarityi
        numpyi = np.stack((num1, num2, num3, np.full(5, np.nan, dtype=object)), axis=0)
        dfi = pd.DataFrame(numpyi, columns=['Title', 'Information', 'Best match', 'Area', 'RF concentration'])
        data_info_i = pd.concat([data_info_i, dfi], axis=0)
    return data_info_i

def save_output1(file_name, output_path, semi_quan_i, data_info_i):
    # Đường dẫn tới file Excel A
    # file_path = f'Data\All pos files\{file_name}.xlsx'

    excel_file = pd.ExcelFile(file_name)

    # Tạo DataFrame từ các sheet hiện tại trừ sheet cuối cùng
    sheets = excel_file.sheet_names[:-1]  # Lấy tất cả sheet trừ sheet cuối cùng
    data_frames = {sheet: pd.read_excel(file_name, sheet_name=sheet) for sheet in sheets}

    # Đường dẫn mới để lưu file Excel
    # new_file_path = f'Data\Results\{file_name}_output1.xlsx'  # Thay thế bằng đường dẫn mới của bạn

    # Lưu các DataFrame và pandas mới vào file Excel mới
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in data_frames.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        # Thêm pandas mới vào các sheet mới
        semi_quan_i.to_excel(writer, sheet_name='Summary 1', index=False)
        data_info_i.to_excel(writer, sheet_name='Summary 2', index=False)

def download_result(file_names=None, sample_count=None):
    st.image("app_demo/image/3_download.png")
    if len(file_names) == 0:
        message = "Failed: No file uploaded for processing"
        st.error(message, icon="❌")
        log_access(message)
    count = 1
    list_output_path = []
    datas_summary = []
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
                similarity_df_i = calculate_similarity_df(filtered_df,target_smiles)
                # print('IR_AREA_: ', IS_area)
                # print(mean)

                semi_quan_i = summary_1(similarity_df_i, filtered_df, file_name.name, IS_area, mean)

                data_info_i = summary_2(semi_quan_i,similarity_df_i,target_smiles)

                output_path = f'app_demo/output/{file_name.name.split(".xlsx")[0]}_output.xlsx'
                list_output_path.append(output_path)
                #Save output 1
                save_output1(file_name, output_path, semi_quan_i,data_info_i)
                datas_summary.append(semi_quan_i)


            except Exception as e:
                message = f"Failed: Error during file processing. Error: {e}"
                st.error(message, icon="❌")
                log_access(message)
        
        count += 1

    if count > 1:
        #Save output 2
        output_2 = pd.concat(datas_summary)
        path_output2 = f'app_demo/output/Semi-quantification_table.xlsx'
        list_output_path.append(path_output2)
        output_2.to_excel(path_output2, index=False)
        st.subheader("Semi-quantification table", divider='rainbow')
        st.dataframe(output_2, height=500, width=1500)
        
        print(list_output_path)
        st.success("The result has been processed and is ready for download!", icon="✅")
        csv = convert_df(output_path)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name_output = f"app_demo/output/SMILE_{current_time}_output.zip"
        file_zip = zip_files(list_output_path, file_name_output)
        log_access(f"Success: Processed and exported results for file /{file_name_output}")
        st.sidebar.subheader("Download Result", divider='rainbow')

        with open(file_zip, "rb") as fp:
            btn = st.sidebar.download_button(
                label="Export to Excel",
                data=fp,
                file_name=f"SMILE_{current_time}_output.zip",
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
