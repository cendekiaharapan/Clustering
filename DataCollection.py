import streamlit as st

# Judul aplikasi
st.title("Clustering Peserta Didik Sekolah Cendekia Harapan")

# Widget untuk mengunggah file
uploaded_file = st.file_uploader("Choose file CSV", type=["csv"])

# Jika pengguna telah mengunggah file
if uploaded_file is not None:
    st.write("Uploaded Files:")
    
    # Menampilkan nama file
    st.write(uploaded_file.name)
    
    # Membaca dan menampilkan isi file
    data = uploaded_file.read()
    st.write("Input File:")
    st.write(data)
