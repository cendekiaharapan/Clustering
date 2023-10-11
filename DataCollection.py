import streamlit as st
import pandas as pd
import openpyxl

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

# Halaman Streamlit
st.title("Clustering App")

uploaded_file = st.file_uploader("Upload a CSV or Excel File:", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    message = verify_file(uploaded_file)
    if "meets the criteria" in message:
        st.success(message)
    else:
        st.error(message)
