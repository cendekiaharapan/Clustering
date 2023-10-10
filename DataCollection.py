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
                return False
            else:
                return True
        elif file_name.endswith(('.xls', '.xlsx')):
            if is_excel_merged(file):
                return False
            else:
                return True
    return False

# Halaman Streamlit
st.title("")  # Judul kosong untuk menyediakan ruang untuk logo

# Tambahkan logo dan judul dalam elemen HTML
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: center;">
        <img src="https://url/logo.png" alt="Logo" style="width: 100px; height: 100px;">
        <h1 style="margin-left: 20px;">Students Clustering at Cendekia Harapan School</h1>
    </div>
    """
    , unsafe_allow_html=True
)

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
                else:
                    st.error(f"File '{uploaded_file.name}' does not meet the criteria")

# Tombol Verify Files
if st.button("Verify Files"):
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            with st.empty():
                if verify_file(uploaded_file):
                    st.success(f"File '{uploaded_file.name}' meets the criteria")
                else:
                    st.error(f"File '{uploaded_file.name}' does not meet the criteria")
    else:
        st.warning("Please upload one or more files first!")
