import streamlit as st
import pandas as pd
import openpyxl
from io import BytesIO
import matplotlib.pyplot as plt
<<<<<<< Updated upstream
=======
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from yellowbrick.cluster import KElbowVisualizer
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import streamlit as st
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
>>>>>>> Stashed changes
import seaborn as sns
import pandas.api.types as ptypes

# Function to preprocess the data based on user selections
def preprocess_data(data, selected_columns, preprocessing_method):
    if preprocessing_method == "Drop":
        data = data.dropna(subset=selected_columns)
    elif preprocessing_method == "Imputation":
        for column in selected_columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                # Check if the numerical column is continuous or categorical
                if len(data[column].unique()) < 0.5 * len(data[column]):
                    # Calculate the mode for categorical numerical columns
                    mode_value = data[column].mode().values[0]
                else:
                    # Calculate the mean for continuous numerical columns
                    mode_value = data[column].mean()
                data[column].fillna(mode_value, inplace=True)
            else:
                # Impute with mode based on the most frequent value for non-numeric columns
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
                df = pd.read_csv(file, encoding='latin1')
                if df.shape[0] < 2 or df.shape[1] < 2:
                    return df, f"File '{file_name}' meets the criteria"
                else:
                    return df, f"File '{file_name}' meets the criteria"
            except Exception as e:
                return None, f"Error: {e}"
        elif file_name.endswith(('.xls', '.xlsx')):
            if is_excel_merged(file):
                return None, f"File '{file_name}' does not meet the criteria"
            else:
                df = pd.read_excel(file)
                if df.empty or df.shape[0] < 2 or df.shape[1] < 2:
                    return None, f"File '{file_name}' does not meet the criteria"
                return df, f"File '{file_name}' meets the criteria"
    return None, "Please upload a file."

st.set_option('deprecation.showPyplotGlobalUse', False)
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

            selected_columns = st.sidebar.multiselect("Select columns to be clustered", data.columns)

            preprocessing_method = st.sidebar.radio("Select preprocessing method", ("Drop", "Imputation"))

            if preprocessing_method in ["Drop", "Imputation"]:
                preprocessed_data = preprocess_data(data.copy(), selected_columns, preprocessing_method)
                st.write(f"Preprocessed Data ({preprocessing_method} Method):")
                st.write(preprocessed_data)
<<<<<<< Updated upstream

                for column in selected_columns:
                    unique_values = preprocessed_data[column].unique()
                    if len(unique_values) == 1:
                        st.success(f"Only one category (value) in column {column}: {unique_values[0]}, and the number of it's number is: {len(preprocessed_data[column])}")
                    elif ptypes.is_numeric_dtype(preprocessed_data[column]):
                        if len(preprocessed_data[column].unique()) < 0.5 * len(preprocessed_data[column]):
                            if len(preprocessed_data[column].unique()) > 5:
                                if len(preprocessed_data[column].unique()) < 45:
                                    plt.figure(figsize=(8, 6))
                                    sns.countplot(y=preprocessed_data[column])
                                    plt.xticks(rotation=0)
                                    st.pyplot()
                                else:
                                    st.error(f"Skipping visualization for column {column} due to too many unique values.")
                            else:
                                plt.figure(figsize=(8, 6))
                                sns.countplot(x=preprocessed_data[column])
                                plt.xticks(rotation=0)
                                st.pyplot()
                        else:
                            plt.figure(figsize=(8, 6))
                            sns.histplot(data=preprocessed_data[column], kde=True)
                            st.pyplot()
                    else:
                        if len(preprocessed_data[column].unique()) > 5:
                            if len(preprocessed_data[column].unique()) < 45:
                                plt.figure(figsize=(8, 6))
                                sns.countplot(y=preprocessed_data[column])
                                plt.xticks(rotation=0)
                                st.pyplot()
                            else:
                                st.error(f"Skipping visualization for column {column} due to too many unique values.")
                        else:
                            plt.figure(figsize=(8, 6))
                            sns.countplot(x=preprocessed_data[column])
                            plt.xticks(rotation=0)
                            st.pyplot()
        else:
            st.error(verification_result)
    else:
        st.error(verification_result)
=======
                
                # Visualize each cluster separately
                for cluster_label in preprocessed_data['Cluster'].unique():
                    st.write(f"Visualizations for Cluster {cluster_label}:")
                    cluster_data = preprocessed_data[preprocessed_data['Cluster'] == cluster_label]
                    for column in selected_columns:
                        unique_values = cluster_data[column].unique()
                        if len(unique_values) == 1:
                            st.success(f"Only one category (value) in column {column} for Cluster {cluster_label}: {unique_values[0]}, and the number of it's number is: {len(cluster_data[column])}")
                        elif ptypes.is_numeric_dtype(cluster_data[column]):
                            if len(cluster_data[column].unique()) < 0.5 * len(cluster_data[column]):
                                if len(cluster_data[column].unique()) > 5:
                                    if len(cluster_data[column].unique()) < 45:
                                        if len(cluster_data[column].unique()) > 20:
                                            plt.figure(figsize=(8, 6))
                                            sns.histplot(data=cluster_data[column], kde=True)
                                            st.pyplot()
                                        else:
                                            plt.figure(figsize=(8, 6))
                                            sns.countplot(y=cluster_data[column])
                                            plt.xticks(rotation=0)
                                            st.pyplot()
                                    else:
                                        st.error(f"Skipping visualization for column {column} in Cluster {cluster_label} due to too many unique values.")
                                else:
                                    plt.figure(figsize=(8, 6))
                                    sns.countplot(x=cluster_data[column])
                                    plt.xticks(rotation=0)
                                    st.pyplot()
                            else:
                                plt.figure(figsize=(8, 6))
                                sns.histplot(data=cluster_data[column], kde=True)
                                st.pyplot()
                        else:
                            if len(cluster_data[column].unique()) > 5:
                                if len(cluster_data[column].unique()) < 45:
                                    plt.figure(figsize=(8, 6))
                                    sns.countplot(y=cluster_data[column])
                                    plt.xticks(rotation=0)
                                    st.pyplot()
                                else:
                                    st.error(f"Skipping visualization for column {column} in Cluster {cluster_label} due to too many unique values.")
                            else:
                                plt.figure(figsize=(8, 6))
                                sns.countplot(x=cluster_data[column])
                                plt.xticks(rotation=0)
                                st.pyplot()
                st.success("Manual Clustering completed!")

            elif len(categorical_columns) == 0:
                st.info("Performing Manual Clustering Using K-Means Algorithm... (This may take a moment)")
                # If there are no categorical columns, perform K-Means clustering
                # Standardize the continuous columns
                scaler = RobustScaler()
                selected_data[continuous_columns] = scaler.fit_transform(selected_data[continuous_columns])
                st.session_state.scaler = scaler
                
                # Create a K-Means model
                kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, random_state=0, verbose=2)

                # Fit the model to the data
                clusters = kmeans.fit_predict(selected_data[continuous_columns])

                st.session_state.model = kmeans

                # Add cluster labels to the preprocessed_data
                preprocessed_data['Cluster'] = clusters

                # Display the clustered data
                st.write("Manual Clustering Result:")
                st.write(f"Number of clusters : {num_clusters}")
                st.write(preprocessed_data)
                
                # Visualize each cluster separately
                for cluster_label in preprocessed_data['Cluster'].unique():
                    st.write(f"Visualizations for Cluster {cluster_label}:")
                    cluster_data = preprocessed_data[preprocessed_data['Cluster'] == cluster_label]
                    for column in selected_columns:
                        unique_values = cluster_data[column].unique()
                        if len(unique_values) == 1:
                            st.success(f"Only one category (value) in column {column} for Cluster {cluster_label}: {unique_values[0]}, and the number of it's number is: {len(cluster_data[column])}")
                        elif ptypes.is_numeric_dtype(cluster_data[column]):
                            if len(cluster_data[column].unique()) < 0.5 * len(cluster_data[column]):
                                if len(cluster_data[column].unique()) > 5:
                                    if len(cluster_data[column].unique()) < 45:
                                        if len(cluster_data[column].unique()) > 20:
                                            plt.figure(figsize=(8, 6))
                                            sns.histplot(data=cluster_data[column], kde=True)
                                            st.pyplot()
                                        else:
                                            plt.figure(figsize=(8, 6))
                                            sns.countplot(y=cluster_data[column])
                                            plt.xticks(rotation=0)
                                            st.pyplot()
                                    else:
                                        st.error(f"Skipping visualization for column {column} in Cluster {cluster_label} due to too many unique values.")
                                else:
                                    plt.figure(figsize=(8, 6))
                                    sns.countplot(x=cluster_data[column])
                                    plt.xticks(rotation=0)
                                    st.pyplot()
                            else:
                                plt.figure(figsize=(8, 6))
                                sns.histplot(data=cluster_data[column], kde=True)
                                st.pyplot()
                        else:
                            if len(cluster_data[column].unique()) > 5:
                                if len(cluster_data[column].unique()) < 45:
                                    plt.figure(figsize=(8, 6))
                                    sns.countplot(y=cluster_data[column])
                                    plt.xticks(rotation=0)
                                    st.pyplot()
                                else:
                                    st.error(f"Skipping visualization for column {column} in Cluster {cluster_label} due to too many unique values.")
                            else:
                                plt.figure(figsize=(8, 6))
                                sns.countplot(x=cluster_data[column])
                                plt.xticks(rotation=0)
                                st.pyplot()
                st.success("Manual Clustering completed!")

            else:
                st.info("Performing Manual Clustering Using K-Prototypes Algorithm... (This may take a moment)")
                # If there are both categorical and continuous columns, perform K-Prototypes clustering
                # Apply one-hot encoding to categorical columns
                selected_data = pd.get_dummies(selected_data, columns=categorical_columns)
                
                # Standardize the continuous columns
                scaler = RobustScaler()
                selected_data[continuous_columns] = scaler.fit_transform(selected_data[continuous_columns])
                st.session_state.scaler = scaler
                
                # Create a K-Prototypes model
                try :
                    kprot = KPrototypes(n_clusters=num_clusters, init='Cao', n_init=1, verbose=2)
                    # Fit the model to the data
                    clusters = kprot.fit_predict(selected_data, categorical=list(range(len(categorical_columns))))
                    st.session_state.model = kprot
                    # Add cluster labels to the preprocessed_data
                    preprocessed_data['Cluster'] = clusters
                    # Display the clustered data
                    st.write("Manual Clustering Result:")
                    st.write(f"Number of clusters : {num_clusters}")
                    st.write(preprocessed_data)
                    
                    # Visualize each cluster separately
                    for cluster_label in preprocessed_data['Cluster'].unique():
                        st.write(f"Visualizations for Cluster {cluster_label}:")
                        cluster_data = preprocessed_data[preprocessed_data['Cluster'] == cluster_label]
                        for column in selected_columns:
                            unique_values = cluster_data[column].unique()
                            if len(unique_values) == 1:
                                st.success(f"Only one category (value) in column {column} for Cluster {cluster_label}: {unique_values[0]}, and the number of it's number is: {len(cluster_data[column])}")
                            elif ptypes.is_numeric_dtype(cluster_data[column]):
                                if len(cluster_data[column].unique()) < 0.5 * len(cluster_data[column]):
                                    if len(cluster_data[column].unique()) > 5:
                                        if len(cluster_data[column].unique()) < 45:
                                            if len(cluster_data[column].unique()) > 20:
                                                plt.figure(figsize=(8, 6))
                                                sns.histplot(data=cluster_data[column], kde=True)
                                                st.pyplot()
                                            else:
                                                plt.figure(figsize=(8, 6))
                                                sns.countplot(y=cluster_data[column])
                                                plt.xticks(rotation=0)
                                                st.pyplot()
                                        else:
                                            st.error(f"Skipping visualization for column {column} in Cluster {cluster_label} due to too many unique values.")
                                    else:
                                        plt.figure(figsize=(8, 6))
                                        sns.countplot(x=cluster_data[column])
                                        plt.xticks(rotation=0)
                                        st.pyplot()
                                else:
                                    plt.figure(figsize=(8, 6))
                                    sns.histplot(data=cluster_data[column], kde=True)
                                    st.pyplot()
                            else:
                                if len(cluster_data[column].unique()) > 5:
                                    if len(cluster_data[column].unique()) < 45:
                                        plt.figure(figsize=(8, 6))
                                        sns.countplot(y=cluster_data[column])
                                        plt.xticks(rotation=0)
                                        st.pyplot()
                                    else:
                                        st.error(f"Skipping visualization for column {column} in Cluster {cluster_label} due to too many unique values.")
                                else:
                                    plt.figure(figsize=(8, 6))
                                    sns.countplot(x=cluster_data[column])
                                    plt.xticks(rotation=0)
                                    st.pyplot()
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
                    st.session_state.model = kmodes

                    # Add cluster labels to the preprocessed_data
                    preprocessed_data['Cluster'] = clusters

                    # Display the clustered data
                    st.write("Automatic Clustering Result:")
                    st.write("Optimal Number Of Cluster:", optimal_num_clusters)
                    st.write(preprocessed_data)
                    
                    # Visualize each cluster separately
                    for cluster_label in preprocessed_data['Cluster'].unique():
                        st.write(f"Visualizations for Cluster {cluster_label}:")
                        cluster_data = preprocessed_data[preprocessed_data['Cluster'] == cluster_label]
                        for column in selected_columns:
                            unique_values = cluster_data[column].unique()
                            if len(unique_values) == 1:
                                st.success(f"Only one category (value) in column {column} for Cluster {cluster_label}: {unique_values[0]}, and the number of it's number is: {len(cluster_data[column])}")
                            elif ptypes.is_numeric_dtype(cluster_data[column]):
                                if len(cluster_data[column].unique()) < 0.5 * len(cluster_data[column]):
                                    if len(cluster_data[column].unique()) > 5:
                                        if len(cluster_data[column].unique()) < 45:
                                            if len(cluster_data[column].unique()) > 20:
                                                plt.figure(figsize=(8, 6))
                                                sns.histplot(data=cluster_data[column], kde=True)
                                                st.pyplot()
                                            else:
                                                plt.figure(figsize=(8, 6))
                                                sns.countplot(y=cluster_data[column])
                                                plt.xticks(rotation=0)
                                                st.pyplot()
                                        else:
                                            st.error(f"Skipping visualization for column {column} in Cluster {cluster_label} due to too many unique values.")
                                    else:
                                        plt.figure(figsize=(8, 6))
                                        sns.countplot(x=cluster_data[column])
                                        plt.xticks(rotation=0)
                                        st.pyplot()
                                else:
                                    plt.figure(figsize=(8, 6))
                                    sns.histplot(data=cluster_data[column], kde=True)
                                    st.pyplot()
                            else:
                                if len(cluster_data[column].unique()) > 5:
                                    if len(cluster_data[column].unique()) < 45:
                                        plt.figure(figsize=(8, 6))
                                        sns.countplot(y=cluster_data[column])
                                        plt.xticks(rotation=0)
                                        st.pyplot()
                                    else:
                                        st.error(f"Skipping visualization for column {column} in Cluster {cluster_label} due to too many unique values.")
                                else:
                                    plt.figure(figsize=(8, 6))
                                    sns.countplot(x=cluster_data[column])
                                    plt.xticks(rotation=0)
                                    st.pyplot()
                    st.success("Automatic Clustering completed!")
                else:
                    st.error("No Optimal Clusters Found, Please Retry")
                
            elif len(categorical_columns) == 0:
                st.info("Performing Automatic Clustering Using K-Means Algorithm... (This may take a moment)")
                # If there are no categorical columns, perform K-Means clustering
                # Standardize the continuous columns
                scaler = RobustScaler()
                selected_data[continuous_columns] = scaler.fit_transform(selected_data[continuous_columns])
                st.session_state.scaler = scaler
                
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
                    st.session_state.model = kmeans

                    # Add cluster labels to the preprocessed_data
                    preprocessed_data['Cluster'] = clusters

                    # Display the clustered data
                    st.write("Automatic Clustering Result:")
                    st.write("Optimal Number Of Cluster:", optimal_num_clusters)
                    st.write(preprocessed_data)
                    
                    for column in selected_columns:
                        unique_values = preprocessed_data[column].unique()
                        if len(unique_values) == 1:
                            st.success(f"Only one category (value) in column {column}: {unique_values[0]}, and the number of it's number is: {len(preprocessed_data[column])}")
                        elif ptypes.is_numeric_dtype(preprocessed_data[column]):
                            if len(preprocessed_data[column].unique()) < 0.5 * len(preprocessed_data[column]):
                                if len(preprocessed_data[column].unique()) > 5:
                                    if len(preprocessed_data[column].unique()) < 45:
                                        if len(preprocessed_data[column].unique()) > 20:
                                            plt.figure(figsize=(8, 6))
                                            sns.histplot(data=preprocessed_data[column], kde=True)
                                            st.pyplot()
                                        else:
                                            plt.figure(figsize=(8, 6))
                                            sns.countplot(y=preprocessed_data[column])
                                            plt.xticks(rotation=0)
                                            st.pyplot()
                                    else:
                                        st.error(f"Skipping visualization for column {column} due to too many unique values.")
                                else:
                                    plt.figure(figsize=(8, 6))
                                    sns.countplot(x=preprocessed_data[column])
                                    plt.xticks(rotation=0)
                                    st.pyplot()
                            else:
                                plt.figure(figsize=(8, 6))
                                sns.histplot(data=preprocessed_data[column], kde=True)
                                st.pyplot()
                        else:
                            if len(preprocessed_data[column].unique()) > 5:
                                if len(preprocessed_data[column].unique()) < 45:
                                    plt.figure(figsize=(8, 6))
                                    sns.c# Visualize each cluster separately
                    for cluster_label in preprocessed_data['Cluster'].unique():
                        st.write(f"Visualizations for Cluster {cluster_label}:")
                        cluster_data = preprocessed_data[preprocessed_data['Cluster'] == cluster_label]
                        for column in selected_columns:
                            unique_values = cluster_data[column].unique()
                            if len(unique_values) == 1:
                                st.success(f"Only one category (value) in column {column} for Cluster {cluster_label}: {unique_values[0]}, and the number of it's number is: {len(cluster_data[column])}")
                            elif ptypes.is_numeric_dtype(cluster_data[column]):
                                if len(cluster_data[column].unique()) < 0.5 * len(cluster_data[column]):
                                    if len(cluster_data[column].unique()) > 5:
                                        if len(cluster_data[column].unique()) < 45:
                                            if len(cluster_data[column].unique()) > 20:
                                                plt.figure(figsize=(8, 6))
                                                sns.histplot(data=cluster_data[column], kde=True)
                                                st.pyplot()
                                            else:
                                                plt.figure(figsize=(8, 6))
                                                sns.countplot(y=cluster_data[column])
                                                plt.xticks(rotation=0)
                                                st.pyplot()
                                        else:
                                            st.error(f"Skipping visualization for column {column} in Cluster {cluster_label} due to too many unique values.")
                                    else:
                                        plt.figure(figsize=(8, 6))
                                        sns.countplot(x=cluster_data[column])
                                        plt.xticks(rotation=0)
                                        st.pyplot()
                                else:
                                    plt.figure(figsize=(8, 6))
                                    sns.histplot(data=cluster_data[column], kde=True)
                                    st.pyplot()
                            else:
                                if len(cluster_data[column].unique()) > 5:
                                    if len(cluster_data[column].unique()) < 45:
                                        plt.figure(figsize=(8, 6))
                                        sns.countplot(y=cluster_data[column])
                                        plt.xticks(rotation=0)
                                        st.pyplot()
                                    else:
                                        st.error(f"Skipping visualization for column {column} in Cluster {cluster_label} due to too many unique values.")
                                else:
                                    plt.figure(figsize=(8, 6))
                                    sns.countplot(x=cluster_data[column])
                                    plt.xticks(rotation=0)
                                    st.pyplot()
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
                st.session_state.scaler = scaler
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
                    st.session_state.model = kprot

                    # Add cluster labels to the preprocessed_data
                    preprocessed_data['Cluster'] = clusters

                    # Display the clustered data
                    
                    
                    st.write("Automatic Clustering Result:")
                    st.write("Optimal Number Of Cluster:", optimal_num_clusters)
                    st.write(preprocessed_data)
                    
                    # Visualize each cluster separately
                    for cluster_label in preprocessed_data['Cluster'].unique():
                        st.write(f"Visualizations for Cluster {cluster_label}:")
                        cluster_data = preprocessed_data[preprocessed_data['Cluster'] == cluster_label]
                        for column in selected_columns:
                            unique_values = cluster_data[column].unique()
                            if len(unique_values) == 1:
                                st.success(f"Only one category (value) in column {column} for Cluster {cluster_label}: {unique_values[0]}, and the number of it's number is: {len(cluster_data[column])}")
                            elif ptypes.is_numeric_dtype(cluster_data[column]):
                                if len(cluster_data[column].unique()) < 0.5 * len(cluster_data[column]):
                                    if len(cluster_data[column].unique()) > 5:
                                        if len(cluster_data[column].unique()) < 45:
                                            if len(cluster_data[column].unique()) > 20:
                                                plt.figure(figsize=(8, 6))
                                                sns.histplot(data=cluster_data[column], kde=True)
                                                st.pyplot()
                                            else:
                                                plt.figure(figsize=(8, 6))
                                                sns.countplot(y=cluster_data[column])
                                                plt.xticks(rotation=0)
                                                st.pyplot()
                                        else:
                                            st.error(f"Skipping visualization for column {column} in Cluster {cluster_label} due to too many unique values.")
                                    else:
                                        plt.figure(figsize=(8, 6))
                                        sns.countplot(x=cluster_data[column])
                                        plt.xticks(rotation=0)
                                        st.pyplot()
                                else:
                                    plt.figure(figsize=(8, 6))
                                    sns.histplot(data=cluster_data[column], kde=True)
                                    st.pyplot()
                            else:
                                if len(cluster_data[column].unique()) > 5:
                                    if len(cluster_data[column].unique()) < 45:
                                        plt.figure(figsize=(8, 6))
                                        sns.countplot(y=cluster_data[column])
                                        plt.xticks(rotation=0)
                                        st.pyplot()
                                    else:
                                        st.error(f"Skipping visualization for column {column} in Cluster {cluster_label} due to too many unique values.")
                                else:
                                    plt.figure(figsize=(8, 6))
                                    sns.countplot(x=cluster_data[column])
                                    plt.xticks(rotation=0)
                                    st.pyplot()
                    st.success("Automatic Clustering completed!")
    
    # Check if categorical_columns exists in session state, and if not, initialize it
    if categorical_columns is not None:
        st.session_state.categorical_columns = categorical_columns  # You can set this to your default value if needed

    # Check if categorical_columns exists in session state, and if not, initialize it
    if continuous_columns is not None:
        st.session_state.continuous_columns = continuous_columns  # You can set this to your default value if needed

    # Check if selected_columns exists in session state, and if not, initialize it
    if selected_columns is not None:
        st.session_state.selected_columns = selected_columns  # You can set this to your default value if needed

    # Check if categorical_columns is defined and selected_columns is defined
    if (st.session_state.categorical_columns is not None and st.session_state.selected_columns is not None):
        categorical_columns = st.session_state.categorical_columns
        selected_columns = st.session_state.selected_columns


    print(st.session_state.categorical_columns, st.session_state.selected_columns)

    if (st.session_state.categorical_columns is not None and st.session_state.selected_columns is not None and st.session_state.model is not None):
        st.sidebar.header("Input Data and Prediction")

        # Create a form for input data
        with st.sidebar.form("Input Data Form"):
            input_data = {}
            for column in st.session_state.selected_columns:
                if column in st.session_state.categorical_columns:
                    input_data[column] = st.text_input(f"Enter {column}", key=f"input_{column}")
                else:
                    input_data[column] = st.number_input(f"Enter {column}", key=f"input_{column}")

            submitted = st.form_submit_button("Predict Cluster")
        
        # Make predictions based on the input data
        if submitted:
            # Convert input data into a DataFrame
            input_df = pd.DataFrame([input_data])

            # Preprocess the input data (similar to how you preprocessed the original data)
            input_df = preprocess_data(input_df, st.session_state.selected_columns, preprocessing_method)

            # Use the clustering model to predict the cluster for the input data
            if len(st.session_state.categorical_columns) == 0:
                input_df = pd.get_dummies(input_df, columns=st.session_state.categorical_columns)
                predicted_cluster = st.session_state.model.predict(input_df)
            elif len(st.session_state.continuous_columns) == 0:
                input_df = scaler.transform(input_df[st.session_state.continuous_columns])
                predicted_cluster = st.session_state.model.predict(input_df[st.session_state.continuous_columns])
            else:
                print("input before", input_df)
                input_df = pd.get_dummies(input_df, columns=st.session_state.categorical_columns)
                print(input_df, st.session_state.categorical_columns)
                input_df[st.session_state.continuous_columns] = st.session_state.scaler.transform(input_df[st.session_state.continuous_columns])
                predicted_cluster = st.session_state.model.predict(input_df, categorical=list(range(len(st.session_state.categorical_columns))))

            st.write("Predicted Cluster:")
            st.write(predicted_cluster[0])
>>>>>>> Stashed changes
