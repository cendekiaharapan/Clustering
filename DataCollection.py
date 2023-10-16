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
                # Impute with mode based on the most frequent value
                mode_value = data[column].mode()[0]
                data[column].fillna(mode_value, inplace=True)
            elif data[column].dtype == 'float64':
                # Calculate the mode for numerical columns
                mode_value = data[column].mode().values[0]
                if not pd.notna(mode_value):
                    mode_value = data[column].mean()
                data[column].fillna(mode_value, inplace=True)
    # Only keep selected columns in the preprocessed data
    data = data[selected_columns]
    return data

# Function to check if the Excel file has merged cells
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
        return True

# Function to verify if the file meets the criteria
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

uploaded_file = st.file_uploader("Choose file CSV or Excel", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    st.write("Uploaded Files:")
    st.write(uploaded_file.name)
    
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    st.write("Input File:")
    st.write(data)
    
    st.sidebar.header("Data Preprocessing")
    
    # Allow users to select columns to be clustered
    selected_columns = st.sidebar.multiselect("Select columns to be clustered", data.columns)
    
    # Allow users to select preprocessing method
    preprocessing_method = st.sidebar.radio("Select preprocessing method", ("Drop", "Imputation"))
    
    if preprocessing_method in ["Drop", "Imputation"]:
        preprocessed_data = preprocess_data(data.copy(), selected_columns, preprocessing_method)
        st.write(f"Preprocessed Data ({preprocessing_method} Method):")
        st.write(preprocessed_data)
