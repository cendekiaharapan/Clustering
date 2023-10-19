import streamlit as st
import pandas as pd
import openpyxl
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import LabelEncoder, StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from kmodes.kprototypes import KPrototypes

# Function to preprocess the data based on user selections
def preprocess_data(data, selected_columns, preprocessing_method):
    if preprocessing_method == "Drop":
        data = data.dropna(subset=selected_columns)
    elif preprocessing_method == "Imputation":
        for column in selected_columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                # Calculate the mode for numerical columns
                mode_value = data[column].mode().values[0]
                if pd.isna(mode_value):
                    mode_value = data[column].mean()
                data[column].fillna(mode_value, inplace=True)
            else:
                # Impute with mode based on the most frequent value
                mode_value = data[column].mode()[0]
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
        if "meets the criteria" in verification_result:
            st.success(verification_result)
            st.sidebar.header("Data Preprocessing")
            
            # Allow users to select columns to be clustered
            selected_columns = st.sidebar.multiselect("Select columns to be clustered", data.columns)
            
            # Allow users to select preprocessing method
            preprocessing_method = st.sidebar.radio("Select preprocessing method", ("Drop", "Imputation"))
            
            st.sidebar.header("Clustering Method")
    # Add radio button for manual or automatic clustering
            clustering_method = st.sidebar.radio("Select Clustering Method", ("Manual", "Automatic"))

            if preprocessing_method in ["Drop", "Imputation"]:
                preprocessed_data = preprocess_data(data.copy(), selected_columns, preprocessing_method)
                st.write(f"Preprocessed Data ({preprocessing_method} Method):")
                st.write(preprocessed_data)
           
           
            if clustering_method == "Manual":
            # Add input field for the number of clusters
                num_clusters = st.sidebar.number_input("Enter the number of clusters:", min_value=2, value=2)
                if st.sidebar.button("Perform Manual Clustering"):
                    # Select the relevant columns for clustering (numeric and categorical)
                    selected_data = preprocessed_data[selected_columns]
                    
                    # Identify categorical columns
                    categorical_columns = selected_data.select_dtypes(include=['object']).columns
                    label_encoder = LabelEncoder()

                    # Label encoding for categorical columns
                    for column in categorical_columns:
                        selected_data[column] = label_encoder.fit_transform(selected_data[column])

                    # Get the indices of categorical columns
                    categorical_indices = [selected_data.columns.get_loc(col) for col in categorical_columns]

                    # Create a K-Prototypes model
                    kprot = KPrototypes(n_clusters=num_clusters, init='Cao', n_init=1, verbose=2)

                    # Fit the model to the data
                    clusters = kprot.fit_predict(selected_data, categorical=categorical_indices)

                    # Add cluster labels to the preprocessed data
                    preprocessed_data['Cluster'] = clusters

                    # Display the clustered data
                    st.write("Manual Clustering Result:")
                    st.write(preprocessed_data)
            

            if clustering_method == "Automatic":
               if st.sidebar.button("Perform Automatic Clustering"):
                # Select the relevant columns for clustering (numeric and categorical)
                selected_data = preprocessed_data[selected_columns]

                # Identify categorical columns
                categorical_columns = selected_data.select_dtypes(include=['object']).columns
                label_encoder = LabelEncoder()

                # Label encoding for categorical columns
                for column in categorical_columns:
                    selected_data[column] = label_encoder.fit_transform(selected_data[column])

                # Get the indices of categorical columns
                categorical_indices = [selected_data.columns.get_loc(col) for col in categorical_columns]

                # Create a Matplotlib figure and axis
                fig, ax = plt.subplots()

                # Create the KElbowVisualizer
                visualizer = KElbowVisualizer(
                    KPrototypes(n_clusters=(1, 11), init='Cao', n_init=1, verbose=2),
                    k=(1, 11),
                )
                # Fit the visualizer with the data
                visualizer.fit(selected_data, categorical=categorical_indices)

                # Display the elbow method graph
                print("test1")
                # Display the elbow method graph using st.pyplot(fig)
                st.pyplot(fig)
                print("test")
                # Get the optimal number of clusters
                optimal_num_clusters = visualizer.elbow_value_

                # Now, use the optimal_num_clusters for K-Prototypes clustering with random initialization
                kprot = KPrototypes(n_clusters=optimal_num_clusters, init='Cao', n_init=1, verbose=2)
                clusters = kprot.fit_predict(selected_data, categorical=categorical_indices)

                # Add cluster labels to the preprocessed_data
                preprocessed_data['Cluster'] = clusters

                # Display the clustered data
                st.write("Automatic Clustering Result:")
                st.write("Optimal Number Of Cluster:", optimal_num_clusters)
                st.write(preprocessed_data)