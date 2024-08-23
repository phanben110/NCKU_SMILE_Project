import streamlit as st
from streamlit_option_menu import option_menu
import re
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from app_demo.src.p1_text_prompt import *
from app_demo.src.p2_visual_prompt_engineer import *
from app_demo.src.p3_visual_prompt import *
from app_demo.src.footer import settingFooter
import warnings
warnings.filterwarnings("ignore")

# Create an option menu for the main menu in the sidebar
st.set_page_config(page_title="Final Project Advanced IIR", page_icon="app_demo/image/logo_csie2.png")
# st.image("image/title_search.png")
st.sidebar.image("app_demo/image/logo_NCKU.jpeg", use_column_width=True)
with st.sidebar:
    selected = option_menu("Main Menu", ["1. Text Prompt", "2. Visual Prompt Engineer", "3. Visual Prompt"],
                           icons=["blockquote-left", "transparency", "images"], menu_icon="bars", default_index=0)
# Based on the selected option, you can display different content in your web application
# page for select icon https://icons.getbootstrap.com/

settingFooter()
if selected == "1. Text Prompt":
    text_prompt()

elif selected == "2. Visual Prompt Engineer":
    visual_prompt_engineer()

elif selected == "3. Visual Prompt":
    visual_prompt()

    