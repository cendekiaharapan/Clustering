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

# Function to check if an Excel file has merged cells
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
        return True  # Assume there is an error, so display the "file doesn't meet criteria" message

# Function to verify if a file meets the criteria for uploading
def verify_file(file):
    if file is not None:
        file_name = file.name
        if file_name.endswith('.csv'):
            df = pd.read_csv(file)
            if df.empty or df.shape != df.dropna().shape:
                return False
            else:
                return True
        elif file_name.endswith(('.xls', '.xlsx')):
            if is_excel_merged(file):
                return False
            else:
                return True
    return False

# Streamlit page title
st.title("Students Clustering at Cendekia Harapan School")

# HTML representation of logo and title
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: center;">
        <img src="https://www.example.com/logo.png" 
        alt="Logo" style="width: 100px; height: 100px;">
        <h1 style="margin-left: 20px;">Students Clustering at Cendekia Harapan School</h1>
    </div>
    """
    , unsafe_allow_html=True
)

# File upload section
uploaded_files = st.file_uploader("Upload one or more CSV or Excel Files:", type=["csv", "xls", "xlsx"], accept_multiple_files=True)

uploaded_file_names = []

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        with st.empty():
            if uploaded_file.name in uploaded_file_names:
                st.error(f"File '{uploaded_file.name}' is a duplicate and will not be processed.")
            else:
                uploaded_file_names.append(uploaded_file.name)
                if verify_file(uploaded_file):
                    st.success(f"File '{uploaded_file.name}' meets the criteria")
                    
                    # Preprocessing options for each uploaded file
                    st.sidebar.header(f"Data Preprocessing for {uploaded_file.name}")
                    
                    # Allow users to select columns to be clustered
                    selected_columns = st.sidebar.multiselect("Select columns to be clustered", df.columns)
                    
                    # Allow users to select preprocessing method
                    preprocessing_method = st.sidebar.radio("Select preprocessing method", ("Drop", "Imputation"))
                    
                    # Preprocess data if the user selected options
                    if st.sidebar.button("Preprocess Data"):
                        if preprocessing_method == "Drop":
                            preprocessed_data = preprocess_data(df.copy(), selected_columns, preprocessing_method)
                            st.write("Preprocessed Data (Dropped Null Values):")
                            st.write(preprocessed_data)
                        elif preprocessing_method == "Imputation":
                            preprocessed_data = preprocess_data(df.copy(), selected_columns, preprocessing_method)
                            st.write("Preprocessed Data (Imputed with Mode for Objects, Mean/Mode for Numerics):")
                            st.write(preprocessed_data)
                    
                else:
                    st.error(f"File '{uploaded_file.name}' does not meet the criteria")
