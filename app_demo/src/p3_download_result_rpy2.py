import streamlit as st
import os
import requests
import numpy as np
import pandas as pd
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import r
import pandas as pd
import io
import json
import warnings
warnings.filterwarnings("ignore") 

# Install necessary R packages if not already installed
utils = importr('utils')
# Load the R libraries
webchem = importr('webchem')
rcdklibs = importr('rcdklibs')
rcdk = importr('rcdk')
fingerprint = importr('fingerprint')


def smiles_to_mol(smiles): 
  return rcdk.parse_smiles(smiles)[0]


def calculate_similarity(smiles1, smiles2):
  mol1 = smiles_to_mol(smiles1)
  mol2 = smiles_to_mol(smiles2)
  fp1 = rcdk.get_fingerprint(mol1, type="circular")
  fp2 = rcdk.get_fingerprint(mol2, type="circular")
  similarity = fingerprint.distance(fp1, fp2, method="tanimoto")
  return similarity[0]


def download_result(file_name=None, ): 
    st.image("app_demo/image/3_download.png") 
    if file_name is not None: 
        df = pd.read_excel(file_name)
        # Check if the DataFrame is empty
        if df.empty:
            st.warning("The uploaded file is empty. Please upload a file with data.")
            return

        filtered_df = df[['Title', 'Area', 'SMILES']] 
        filtered_df.head()
        filtered_df = filtered_df.dropna() 

        smiles_list = list(filtered_df["SMILES"]) 

        # target_smiles = [
        #     "C(C(F)(F)F)(OC(C(F)(F)S(=O)(=O)O)(F)F)(F)F",
        #                 "C(C(C(C(C(F)(F)Cl)(F)F)(F)F)(F)F)(C(C(C(OC(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(F)F)(F)F",
        #                 "C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(C(C(F)(F)F)(F)F)(F)F",
        #                 "C(C(C(C(F)(F)Cl)(F)F)(F)F)(C(C(OC(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(F)F",
        #                 "C(C(C(C(F)(F)F)(F)F)(F)F)(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F",
        #                 "C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F"
        # ]

        with open('config.json', 'r') as file:
            config = json.load(file)

        # Extract specific values
        IS_area = config.get('IS_area')
        mean = config.get('mean')
        target_smiles = config.get('target_smiles')

        # Tạo ma trận tương đồng
        similarity_matrix = np.zeros((len(smiles_list), len(target_smiles)))
        for i, smiles1 in enumerate(smiles_list):
            for j, smiles2 in enumerate(target_smiles):
                similarity_matrix[i, j] = calculate_similarity(smiles1, smiles2)

        # Chuyển kết quả thành định dạng DataFrame
        similarity_df = pd.DataFrame(similarity_matrix, columns=target_smiles, index=smiles_list) 

        ## Create Semi_quantification
        #Get Sample column
        file_name_excel = file_name.name
        try: 
            samplei = file_name_excel.split('_')[0]
        except:
            samplei = file_name.name.split(".xlsx")[0] 

        sample = np.full((len(smiles_list),), samplei)
        semi_quan = pd.DataFrame({'Sample': sample})
        #Add Title column
        semi_quan['Title'] = filtered_df['Title'].values
        #Add Best match column
        semi_quan['Best match'] = np.argmax(similarity_df.values, axis=1) + 1
        #Add Area column
        semi_quan['Area'] = filtered_df['Area'].values
    
        #Add RF concerntration column

        RF = []
        for i in range(len(semi_quan)):
            rfi = semi_quan['Area'].values[i]/(IS_area*mean[semi_quan['Best match'].values[i]-1])*40
            RF.append(rfi)
        semi_quan['RF concerntration'] = RF

        ## Create Data_information
        data_info = pd.DataFrame()
        for i in range(len(smiles_list)):
            num1 = [semi_quan['Title'].values[i], str('SMILES: ') + similarity_df.index[i], semi_quan['Best match'].values[i],semi_quan['Area'].values[i],semi_quan['RF concerntration'].values[i]]
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

        #Save Data
        st.subheader("Sheet 1: Semi-quantitation", divider='rainbow')
        st.dataframe(semi_quan, height=350, width=1500)
        st.subheader("Sheet 2: Addition Output", divider='rainbow')
        st.dataframe(data_info, height=350, width=1500)

        output_path = f'app_demo/output/{file_name.name.split(".xlsx")[0]}_output.xlsx'
        print(output_path)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            semi_quan.to_excel(writer, sheet_name='Sheet1', index=False)
            data_info.to_excel(writer, sheet_name='Sheet2', index=False) 
        
        csv = convert_df(output_path)
        st.sidebar.subheader("Download Result", divider='rainbow')


        # Initialize session state for tracking the button click
        if 'download_clicked' not in st.session_state:
            st.session_state.download_clicked = False

        
        st.sidebar.download_button(
            label="Export to Excel",
            data=csv,
            file_name= file_name.name.split(".xlsx")[0] + "_output.xlsx"
        )
   
    elif file_name is None:
        st.error("Please upload a data file to start processing.", icon="❌")

def convert_df(file_excel):
    # Read all sheets into a dictionary of DataFrames
    dfs = pd.read_excel(file_excel, sheet_name=None)
    
    # Create a BytesIO buffer to hold the combined Excel file in memory
    output = io.BytesIO()
    
    # Write each DataFrame to a separate sheet in the Excel file
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Get the binary content of the Excel file
    excel_data = output.getvalue()
    
    # Close the buffer
    output.close()
    return excel_data
    

