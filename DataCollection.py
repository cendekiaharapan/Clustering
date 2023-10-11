import streamlit as st
import pandas as pd
import openpyxl

# Function to preprocess the data based on user selections
def preprocess_data(data, selected_columns, preprocessing_method):
    if preprocessing_method == "Drop":
        data = data.dropna(subset=selected_columns)
    elif preprocessing_method == "Imputation":
        for column in selected_columns:
            if data[column].dtype == 'object':
                data[column].fillna(data[column].mode()[0], inplace=True)
            else:
                if len(data[column].unique()) > len(data) * 0.1:
                    data[column].fillna(data[column].mean(), inplace=True)
                else:
                    data[column].fillna(data[column].mode()[0], inplace=True)
    return data

# Fungsi untuk memeriksa apakah file Excel memiliki sel yang digabung (merged)
def is_excel_merged(file_path):
    try:
        workbook = openpyxl.load_workbook(file_path, read_only=True)
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            for merged_cells in sheet.merged_cells.ranges:
                if merged_cells:
                    return True
        return False
    except Exception as e:
        return True  # Anggap saja terdapat error, sehingga munculkan pesan "file tidak memenuhi kriteria"

# Fungsi untuk memeriksa apakah file CSV atau Excel memenuhi kriteria
def verify_file(file):
    if file is not None:
        file_name = file.name
        if file_name.endswith('.csv'):
            df = pd.read_csv(file)
            if df.empty or df.shape != df.dropna().shape:
                return f"File '{file_name}' does not meet the criteria"
            else:
                return f"File '{file_name}' meets the criteria"
        elif file_name.endswith(('.xls', '.xlsx')):
            if is_excel_merged(file):
                return f"File '{file_name}' does not meet the criteria"
            else:
                return f"File '{file_name}' meets the criteria"
    return "Please upload a file."

st.title("Clustering Peserta Didik Sekolah Cendekia Harapan")

uploaded_file = st.file_uploader("Choose file CSV", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    st.write("Uploaded Files:")
    st.write(uploaded_file.name)
    
    data = pd.read_csv(uploaded_file)
    
    st.write("Input File:")
    st.write(data)
    
    st.sidebar.header("Data Preprocessing")
    
    # Allow users to select columns to be clustered
    selected_columns = st.sidebar.multiselect("Select columns to be clustered", data.columns)
    
    # Allow users to select preprocessing method
    preprocessing_method = st.sidebar.radio("Select preprocessing method", ("Drop", "Imputation"))
    
    if preprocessing_method == "Drop":
        preprocessed_data = preprocess_data(data.copy(), selected_columns, preprocessing_method)
        st.write("Preprocessed Data (Dropped Null Values):")
        st.write(preprocessed_data)
    
    elif preprocessing_method == "Imputation":
        preprocessed_data = preprocess_data(data.copy(), selected_columns, preprocessing_method)
        st.write("Preprocessed Data (Imputed with Mode for Objects, Mean/Mode for Numerics):")
        st.write(preprocessed_data)
