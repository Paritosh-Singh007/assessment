import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.title('ResoluteAI Assessment')
# Function for Task 1
def task_1():
    st.title('Task 1: K-Means Clustering')

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file for Task 1", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if 'Cluster' in df.columns:
            df = df.drop(columns=['Cluster'])

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        # Elbow method
        inertia = []
        K = range(1, 11)
        for k in K:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(scaled_data)
            inertia.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(K, inertia, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        st.pyplot(plt)

        # Assume the optimal k is 3
        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=0).fit(scaled_data)

        df['Cluster'] = kmeans.labels_

        st.write("Data with Cluster Labels")
        st.dataframe(df)
        
        # Save clustered data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Clustered Data as CSV", data=csv, file_name='clustered_data.csv', mime='text/csv')

        # Predict new data point
        feature_names = df.drop(columns=['Cluster']).columns
        new_data_point = [st.number_input(f'Input value for {feature}', value=0.0) for feature in feature_names]
        
        if st.button('Predict Cluster'):
            new_data_point_scaled = scaler.transform([new_data_point])
            cluster_label = kmeans.predict(new_data_point_scaled)[0]
            centroids = kmeans.cluster_centers_
            distances = np.linalg.norm(centroids - new_data_point_scaled, axis=1)
            explanation = f"Data point belongs to cluster {cluster_label} because it is closest to centroid {centroids[cluster_label]} with a distance of {distances[cluster_label]:.2f}."
            st.write(f"Cluster Label: {cluster_label}")
            st.write(f"Explanation: {explanation}")

# Function for Task 2
def task_2():
    st.title('Task 2: Random Forest Classifier')

    train_file = st.file_uploader("Choose a training Excel file for Task 2", type="xlsx", key="train")
    test_file = st.file_uploader("Choose a testing Excel file for Task 2", type="xlsx", key="test")

    if train_file is not None and test_file is not None:
        train_df = pd.read_excel(train_file)
        test_df = pd.read_excel(test_file)
        
        # Preprocess data
        X_train = train_df.drop('target', axis=1)  # Replace 'target' with your actual target column name
        y_train = train_df['target']  # Replace 'target' with your actual target column name

        # Split train data for validation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Initialize and train the model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        train_accuracy = model.score(X_train, y_train)
        val_accuracy = model.score(X_val, y_val)
        st.write(f'Training Accuracy: {train_accuracy}')
        st.write(f'Validation Accuracy: {val_accuracy}')

        # Predict on the test set
        test_predictions = model.predict(test_df)

        # Save predictions to a CSV file
        test_predictions_df = pd.DataFrame(test_predictions, columns=['T2'])  # Replace 'T2' with your column name
        csv = test_predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Test Predictions as CSV", data=csv, file_name='test_predictions.csv', mime='text/csv')

# Function for Task 3
def task_3():
    st.title('Task 3: Activity Duration and Count Summary')

    uploaded_file = st.file_uploader("Choose a CSV file for Task 3", type="csv", key="task3")

    if uploaded_file is not None:
        # Load and process the data
        raw_data = pd.read_csv(uploaded_file)
        raw_data['datetime'] = pd.to_datetime(raw_data['date'] + ' ' + raw_data['time'])
        raw_data['position'] = raw_data['position'].str.lower()
        raw_data['duration'] = 1
        raw_data['date'] = raw_data['datetime'].dt.date

        duration_summary = raw_data.groupby(['date', 'position'])['duration'].sum().unstack(fill_value=0)
        activity_count = raw_data.groupby(['date', 'activity']).size().unstack(fill_value=0)
        summary = pd.concat([duration_summary, activity_count], axis=1).fillna(0)
        summary.reset_index(inplace=True)

        st.write("Summary of Activities")
        st.dataframe(summary)

        # Option to download the summary as CSV
        csv = summary.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Summary as CSV", data=csv, file_name='summary.csv', mime='text/csv')

# Sidebar for navigation
st.sidebar.image('images.jpg',width= 150)
st.sidebar.title("Navigation")
task = st.sidebar.radio("Go to", ("Task 1", "Task 2", "Task 3"))

# Display the selected page
if task == "Task 1":
    task_1()
elif task == "Task 2":
    task_2()
elif task == "Task 3":
    task_3()
