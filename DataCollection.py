import streamlit as st
import pandas as pd
import openpyxl
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from yellowbrick.cluster import KElbowVisualizer
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

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

st.title("Clustering App")

uploaded_file = st.file_uploader("Upload a CSV or Excel File:", type=["csv", "xls", "xlsx"])

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
    
    st.sidebar.header("Clustering Method")
    # Add radio button for manual or automatic clustering
    clustering_method = st.sidebar.radio("Select Clustering Method", ("Manual", "Automatic"))
    
    if preprocessing_method in ["Drop", "Imputation"]:
        preprocessed_data = preprocess_data(data.copy(), selected_columns, preprocessing_method)
        st.write(f"Preprocessed Data ({preprocessing_method} Method):")
        st.write(preprocessed_data)

    if clustering_method == "Manual":
        num_clusters = st.sidebar.number_input("Enter the number of clusters:", min_value=2, value=2)


        if st.sidebar.button("Perform Manual Clustering"):
            # Select the relevant columns for clustering (numeric and categorical)
            selected_data = preprocessed_data[selected_columns]

            # Identify categorical and continuous columns
            categorical_columns = selected_data.select_dtypes(include=['object']).columns
            continuous_columns = selected_data.select_dtypes(exclude=['object']).columns

            if len(continuous_columns) == 0:
                st.info("Performing Manual Clustering Using K-Modes Algorithm... (This may take a moment)")
                # If there are no continuous columns, perform K-Modes clustering
                # Apply one-hot encoding to categorical columns
                selected_data = pd.get_dummies(selected_data, columns=categorical_columns)
                
                # Create a K-Modes model
                kmodes = KModes(n_clusters=num_clusters, init='Cao', n_init=1, verbose=2)

                # Fit the model to the data
                clusters = kmodes.fit_predict(selected_data)

                # Add cluster labels to the preprocessed_data
                preprocessed_data['Cluster'] = clusters

                # Display the clustered data
                st.write("Manual Clustering Result:")
                st.write(f"Number of clusters : {num_clusters}")
                st.write(preprocessed_data)
                st.success("Manual Clustering completed!")

            elif len(categorical_columns) == 0:
                st.info("Performing Manual Clustering Using K-Means Algorithm... (This may take a moment)")
                # If there are no categorical columns, perform K-Means clustering
                # Standardize the continuous columns
                scaler = RobustScaler()
                selected_data[continuous_columns] = scaler.fit_transform(selected_data[continuous_columns])
                
                # Create a K-Means model
                kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, random_state=0, verbose=2)

                # Fit the model to the data
                clusters = kmeans.fit_predict(selected_data[continuous_columns])

                # Add cluster labels to the preprocessed_data
                preprocessed_data['Cluster'] = clusters

                # Display the clustered data
                st.write("Manual Clustering Result:")
                st.write(f"Number of clusters : {num_clusters}")
                st.write(preprocessed_data)
                st.success("Manual Clustering completed!")

            else:
                st.info("Performing Manual Clustering Using K-Prototypes Algorithm... (This may take a moment)")
                # If there are both categorical and continuous columns, perform K-Prototypes clustering
                # Apply one-hot encoding to categorical columns
                selected_data = pd.get_dummies(selected_data, columns=categorical_columns)
                
                # Standardize the continuous columns
                scaler = RobustScaler()
                selected_data[continuous_columns] = scaler.fit_transform(selected_data[continuous_columns])
                
                # Create a K-Prototypes model
                try :
                    kprot = KPrototypes(n_clusters=num_clusters, init='Cao', n_init=1, verbose=2)
                    # Fit the model to the data
                    clusters = kprot.fit_predict(selected_data, categorical=list(range(len(categorical_columns))))
                    # Add cluster labels to the preprocessed_data
                    preprocessed_data['Cluster'] = clusters
                    # Display the clustered data
                    st.write("Manual Clustering Result:")
                    st.write(f"Number of clusters : {num_clusters}")
                    st.write(preprocessed_data)
                    st.success("Manual Clustering completed!")
                    
                except: 
                    st.error(f"Cannot perfom cluster in {num_clusters} clusters, Please try another amount of clusters")
            

         


    if clustering_method == "Automatic":
        if st.sidebar.button("Perform Automatic Clustering"):
            # Select the relevant columns for clustering (numeric and categorical)
            selected_data = preprocessed_data[selected_columns]

            # Identify categorical columns
            categorical_columns = selected_data.select_dtypes(include=['object']).columns
            continuous_columns = selected_data.select_dtypes(exclude=['object']).columns

            if len(continuous_columns) == 0:
                st.info("Performing Automatic Clustering Using K-Modes Algorithm... (This may take a moment)")
                selected_data = pd.get_dummies(selected_data, columns=categorical_columns)
                
                fig, ax = plt.subplots()

                for k in range(11, 3, -1):
                    try:
                        # Create the KElbowVisualizer for K-Modes
                        visualizer = KElbowVisualizer(
                            KModes(n_clusters=k, init='Cao', n_init=1, verbose=2),
                            k=(2, k)
                        )
                        # Fit the visualizer with the data
                        visualizer.fit(selected_data)
                    except:
                        print(f"Failed To Cluster For {k}")
                        continue
                    break

                # Display the elbow method graph using st.pyplot(fig)
                st.pyplot(fig)

                # Get the optimal number of clusters
                optimal_num_clusters = visualizer.elbow_value_

                if optimal_num_clusters is not None:
                    # Now, use the optimal_num_clusters for K-Modes clustering
                    kmodes = KModes(n_clusters=optimal_num_clusters, init='Cao', n_init=1, verbose=2)
                    clusters = kmodes.fit_predict(selected_data)

                    # Add cluster labels to the preprocessed_data
                    preprocessed_data['Cluster'] = clusters

                    # Display the clustered data
                    st.write("Automatic Clustering Result:")
                    st.write("Optimal Number Of Cluster:", optimal_num_clusters)
                    st.write(preprocessed_data)
                    st.success("Automatic Clustering completed!")
                else:
                    st.error("No Optimal Clusters Found, Please Retry")
                
            elif len(categorical_columns) == 0:
                st.info("Performing Automatic Clustering Using K-Means Algorithm... (This may take a moment)")
                # If there are no categorical columns, perform K-Means clustering
                # Standardize the continuous columns
                scaler = RobustScaler()
                selected_data[continuous_columns] = scaler.fit_transform(selected_data[continuous_columns])
                
                fig, ax = plt.subplots()

                for k in range(11, 3, -1):
                    try:
                        # Create the KElbowVisualizer for K-Means
                        visualizer = KElbowVisualizer(
                            KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0, verbose=2),
                            k=(2, k)
                        )
                        # Fit the visualizer with the data
                        visualizer.fit(selected_data)
                    except:
                        print(f"Failed To Cluster For {k}")
                        continue
                    break

                # Display the elbow method graph using st.pyplot(fig)
                st.pyplot(fig)

                # Get the optimal number of clusters
                optimal_num_clusters = visualizer.elbow_value_
                if optimal_num_clusters is not None:
                    # Create a K-Means model
                    kmeans = KMeans(n_clusters=optimal_num_clusters, init='k-means++', n_init=10, random_state=0, verbose=2)
                    # Fit the model to the data
                    clusters = kmeans.fit_predict(selected_data[continuous_columns])

                    # Add cluster labels to the preprocessed_data
                    preprocessed_data['Cluster'] = clusters

                    # Display the clustered data
                    st.write("Automatic Clustering Result:")
                    st.write("Optimal Number Of Cluster:", optimal_num_clusters)
                    st.write(preprocessed_data)
                    st.success("Automatic Clustering completed!")
                else:
                    st.error("No Optimal Clusters Found, Please Retry")
            else :        
                st.info("Performing Automatic Clustering Using K-Prototypes Algorithm... (This may take a moment)")
                # Apply one-hot encoding to categorical columns
                selected_data = pd.get_dummies(selected_data, columns=categorical_columns)
                
                # Scale the continuous columns
                scaler = RobustScaler()
                selected_data[continuous_columns] = scaler.fit_transform(selected_data[continuous_columns])
                # Create a Matplotlib figure and axis
                fig, ax = plt.subplots()

                for k in range (11,3,-1) :
                        try :
                            # Create the KElbowVisualizer
                            visualizer = KElbowVisualizer(
                                KPrototypes( init='Cao', n_init=1, verbose=2),
                                k=(2, k),
                            )
                            # Fit the visualizer with the data
                            visualizer.fit(selected_data, categorical=list(range(len(categorical_columns))))
                        except :
                            print(f"Failed To Cluster For {k}")
                            continue
                        break
                

                # Display the elbow method graph using st.pyplot(fig)
                st.pyplot(fig)

                # Get the optimal number of clusters
                
                optimal_num_clusters = visualizer.elbow_value_
                if optimal_num_clusters is None :
                    st.error("No Optimal Clusters Found, Please Retry")
                else :
                    # Now, use the optimal_num_clusters for K-Prototypes clustering with random initialization
                    kprot = KPrototypes(n_clusters=optimal_num_clusters, init='Cao', n_init=1, verbose=2)
                    clusters = kprot.fit_predict(selected_data, categorical=list(range(len(categorical_columns))))

                    # Add cluster labels to the preprocessed_data
                    preprocessed_data['Cluster'] = clusters

                    # Display the clustered data
                    
                    
                    st.write("Automatic Clustering Result:")
                    st.write("Optimal Number Of Cluster:", optimal_num_clusters)
                    st.write(preprocessed_data)
                    st.success("Automatic Clustering completed!")

                