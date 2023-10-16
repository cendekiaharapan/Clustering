import streamlit as st
import pandas as pd
import openpyxl
from io import BytesIO

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
                if pd.isna(mode_value):
                    mode_value = data[column].mean()
                data[column].fillna(mode_value, inplace=True)
    # Only keep selected columns in the preprocessed data
    data = data[selected_columns]
    return data

# Function to check if the Excel file has merged cells
def is_excel_merged(file):
    try:
        file.seek(0)
        content = BytesIO(file.read())
        workbook = openpyxl.load_workbook(content, read_only=True)
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            for merged_cells in sheet.merged_cells.ranges:
                if merged_cells:
                    return True
        return False
    except Exception as e:
        return True  # Assume an error, and show the message "File does not meet the criteria"

# Function to check if the uploaded file meets the criteria
def verify_file(file):
    if file is not None:
        file_name = file.name
        if file_name.endswith('.csv'):
            try:
                df = pd.read_csv(file)
                if df.empty or df.shape != df.dropna().shape:
                    return None, f"File '{file_name}' does not meet the criteria"
                else:
                    return df, f"File '{file_name}' meets the criteria"
            except Exception as e:
                return None, f"Error: {e}"
        elif file_name.endswith(('.xls', '.xlsx')):
            if is_excel_merged(file):
                return None, f"File '{file_name}' does not meet the criteria"
            else:
                df = pd.read_excel(file)
                return df, f"File '{file_name}' meets the criteria"
    return None, "Please upload a file."

st.title("Clustering Peserta Didik Sekolah Cendekia Harapan")

uploaded_file = st.file_uploader("Choose file CSV", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    st.write("Uploaded Files:")
    st.write(uploaded_file.name)
    
    data, verification_result = verify_file(uploaded_file)
    
    if data is not None:
        st.write("Input File:")
        st.write(data)
        st.write(verification_result)
        
        st.sidebar.header("Data Preprocessing")
        
        # Allow users to select columns to be clustered
        selected_columns = st.sidebar.multiselect("Select columns to be clustered", data.columns)
        
        # Allow users to select preprocessing method
        preprocessing_method = st.sidebar.radio("Select preprocessing method", ("Drop", "Imputation"))
        
        if preprocessing_method in ["Drop", "Imputation"]:
            preprocessed_data = preprocess_data(data.copy(), selected_columns, preprocessing_method)
            st.write(f"Preprocessed Data ({preprocessing_method} Method):")
            st.write(preprocessed_data)
    else:
        st.write(verification_result)
