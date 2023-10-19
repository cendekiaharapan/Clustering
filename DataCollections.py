import streamlit as st
import pandas as pd
import openpyxl
import numpy as np
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import LabelEncoder, StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from kmodes.kprototypes import KPrototypes
import matplotlib
matplotlib.use("Agg")

      

# def find_optimal_clusters(data, min_clusters, max_clusters):
#     best_score = -1
#     best_clusters = None

#     for k in range(min_clusters, max_clusters + 1):
#         clusterer = HDBSCAN(min_cluster_size=k, gen_min_span_tree=True)
#         cluster_labels = clusterer.fit_predict(data)

#         if len(set(cluster_labels)) > 1:
#             silhouette_avg = silhouette_samples(data, cluster_labels).mean()
#             if silhouette_avg > best_score:
#                 best_score = silhouette_avg
#                 best_clusters = k

#     return best_clusters

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