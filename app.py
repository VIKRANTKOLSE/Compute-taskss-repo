import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN

#Generating database
def make_datset(no_of_samples=400,centers=3,cluster_std=0.7):
    X,y=make_blobs(n_samples=no_of_samples,centers=centers,cluster_std=cluster_std,random_state=42)
    df=pd.DataFrame(X,columns=['Feature_1','Feature_2'])
    return df,y
X,y=make_datset()


#Page configuration
st.set_page_config(layout="wide", page_title="Clustering Algorithm Experimenter")
st.markdown("Use the sidebar to select a scaling method and clustering algorithm")

#Taking user inputs
with st.sidebar:
    st.header("1. Data Preprocessing")
    
    # Scaling options
    scaling_option = st.selectbox(
        'Select Feature Scaler',
        ('None', 'StandardScaler', 'MinMaxScaler', 'RobustScaler')
    )
    
    #Clustering options
    st.header("2. Clustering Configuration")

    cluster_algo = st.selectbox(
        'Select Clustering Algorithm',
        ('KMeans', 'AgglomerativeClustering', 'DBSCAN')
    )

    if cluster_algo == 'KMeans' or cluster_algo == 'AgglomerativeClustering':
        n_clusters = st.slider('Number of Clusters (k)', min_value=2, max_value=10, value=4, step=1)
    
    elif cluster_algo == 'DBSCAN':
        dbscan_eps = st.slider('Epsilon (eps)', min_value=0.1, max_value=0.5, value=0.25, step=0.01)
        dbscan_min_samples = st.slider('Min Samples', min_value=2, max_value=10, value=5, step=1)


#Applying scaling
def scaling(df, type_of_scaling):
    if type_of_scaling == 'None':
        return df
    
    data_array = df.values
    
    if type_of_scaling == 'StandardScaler':
        scaler = StandardScaler()
    elif type_of_scaling == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif type_of_scaling == 'RobustScaler':
        scaler = RobustScaler()
    
    scaled_data = scaler.fit_transform(data_array)
    return pd.DataFrame(scaled_data, columns=df.columns)

scaled_df=scaling(X,scaling_option)



#Using Clustering algorithm

def perform_clustering(X, algo, n_clusters, eps, min_samples):
    if algo == 'KMeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X)
    
    elif algo == 'AgglomerativeClustering':
        model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        labels = model.fit_predict(X)

    elif algo == 'DBSCAN':
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
    return labels

if cluster_algo == 'DBSCAN':
    cluster_labels = perform_clustering(scaled_df, cluster_algo, None, dbscan_eps, dbscan_min_samples)
else:
    cluster_labels = perform_clustering(scaled_df, cluster_algo, n_clusters, None, None)


#Evaluating the outputs
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score
s_score = silhouette_score(scaled_df, cluster_labels)
dbi_score = davies_bouldin_score(scaled_df, cluster_labels)
ch_score = calinski_harabasz_score(scaled_df, cluster_labels)


#Plotting the clusters
plot_df = scaled_df.copy()
plot_df['Cluster'] = cluster_labels
valid_cluster_labels = np.unique(cluster_labels)

st.subheader(f"Visualization: {scaling_option} + {cluster_algo}")

st.scatter_chart(plot_df,x='Feature_1',y='Feature_2',color='Cluster',height=500)