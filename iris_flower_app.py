# -*- coding: utf-8 -*-
"""
Iris Flower Clustering Dashboard - Redesigned Edition
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris

# -------------------------------------------------------------
# 1. Page Configuration & Custom UI Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="Iris K-Means Insights", 
    page_icon="🌸", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern, elegant CSS styling (Corrected parameter here)
st.markdown("""
    <style>
    /* Main body adjustments */
    .main {
        background-color: #0e1117;
        font-family: 'Inter', sans-serif;
    }
    /* Dynamic title styling */
    .title-text {
        font-weight: 800;
        background: linear-gradient(45deg, #FF4B4B, #4B79FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    .subtitle-text {
        color: #a3a8b4;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    /* Card design for data stats */
    .metric-card {
        background-color: #1e222b;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2d3139;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Hero Header
# -------------------------------------------------------------
st.markdown('<p class="title-text">🌸 Iris Cluster Studio</p>', unsafe_with_html=True) if False else st.markdown('<p class="title-text">🌸 Iris Cluster Studio</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">An interactive machine learning sandbox exploring K-Means Clustering on the classic Iris dataset features.</p>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 3. Data Loading & Preparation
# -------------------------------------------------------------
@st.cache_data
def load_data():
    iris = load_iris()
    df_raw = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_raw['flower'] = iris.target
    
    # Drop sepal features and target for simplicity as per exercise instructions
    df_filtered = df_raw.drop(['sepal length (cm)', 'sepal width (cm)', 'flower'], axis='columns')
    return df_filtered

df = load_data()

# -------------------------------------------------------------
# 4. Sidebar Layout & Control Panels
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1526047932273-341f2a7631f9?auto=format&fit=crop&w=500&q=80", use_container_width=True)
    st.markdown("### ⚙️ Engine Parameters")
    
    use_scaling = st.checkbox(
        "⚡ Apply MinMaxScaler", 
        value=True,  # Default to True for better ML practices!
        help="Scaling normalizes features to a shared range (0-1), preventing features with larger magnitudes from dominating."
    )
    
    k_value = st.slider(
        "🎯 Number of Clusters (K)", 
        min_value=1, 
        max_value=6, 
        value=3,
        help="Choose how many cluster groups the model should segment the flowers into."
    )
    
    st.divider()
    st.markdown("✨ *Tip: Toggle the scaler to watch how the cluster boundaries bend and adjust in real-time.*")

# -------------------------------------------------------------
# 5. Data Preprocessing (Scaling)
# -------------------------------------------------------------
if use_scaling:
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[['petal length (cm)', 'petal width (cm)']])
    df_cluster = pd.DataFrame(scaled_features, columns=['petal length (cm)', 'petal width (cm)'])
else:
    df_cluster = df.copy()

# -------------------------------------------------------------
# 6. K-Means Clustering Core
# -------------------------------------------------------------
km = KMeans(n_clusters=k_value, random_state=42, n_init='auto')
yp = km.fit_predict(df_cluster)
df_cluster['Cluster'] = [f"Cluster {i}" for i in yp] # Convert to clean string categories for nicer chart legends

# -------------------------------------------------------------
# 7. Mini KPI Summary Grid
# -------------------------------------------------------------
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.markdown(f'<div class="metric-card"><span style="color:#a3a8b4; font-size:0.9rem;">Target Clusters (K)</span><br><b style="font-size:1.8rem; color:#FF4B4B;">{k_value}</b></div>', unsafe_allow_html=True)
with m_col2:
    status = "Active" if use_scaling else "Inactive"
    color = "#4B79FF" if use_scaling else "#a3a8b4"
    st.markdown(f'<div class="metric-card"><span style="color:#a3a8b4; font-size:0.9rem;">MinMax Scaler</span><br><b style="font-size:1.8rem; color:{color};">{status}</b></div>', unsafe_allow_html=True)
with m_col3:
    st.markdown(f'<div class="metric-card"><span style="color:#a3a8b4; font-size:0.9rem;">Total Data Points</span><br><b style="font-size:1.8rem; color:#10B981;">{len(df_cluster)}</b></div>', unsafe_allow_html=True)

st.write("")

# -------------------------------------------------------------
# 8. Interactive Dashboards (Two Column Layout)
# -------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📊 Interactive Cluster Map")
    
    # Custom vibrant color palette for modern design
    custom_colors = ["#4B79FF", "#FF4B4B", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899"]
    
    fig = px.scatter(
        df_cluster, 
        x='petal length (cm)', 
        y='petal width (cm)', 
        color='Cluster',
        color_discrete_sequence=custom_colors,
        labels={'petal length (cm)': 'Petal Length', 'petal width (cm)': 'Petal Width'},
        template="plotly_dark",
        opacity=0.85
    )
    
    # Custom visual styling adjustments for the plot
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='white')), selector=dict(mode='markers'))
    
    # Draw Centroids elegantly if K > 1
    if k_value > 1:
        centroids_df = pd.DataFrame(km.cluster_centers_, columns=['petal length (cm)', 'petal width (cm)'])
        fig.add_trace(
            go.Scatter(
                x=centroids_df['petal length (cm)'],
                y=centroids_df['petal width (cm)'],
                mode='markers',
                marker=dict(color='white', size=15, symbol='x', line=dict(width=2, color='black')),
                name='Centroids',
                showlegend=True
            )
        )
        
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 📐 Elbow Evaluation Plot")
    
    # Dynamic calculations for the Elbow Plot
    sse = []
    k_rng = range(1, 10)
    for k in k_rng:
        km_loop = KMeans(n_clusters=k, random_state=42, n_init='auto')
        km_loop.fit(df_cluster[['petal length (cm)', 'petal width (cm)']])
        sse.append(km_loop.inertia_)
        
    elbow_df = pd.DataFrame({'K': list(k_rng), 'SSE': sse})
    
    fig_elbow = px.line(
        elbow_df, 
        x='K', 
        y='SSE', 
        markers=True,
        template="plotly_dark",
        labels={'SSE': 'Sum of Squared Error (Inertia)'}
    )
    
    fig_elbow.update_traces(line=dict(color='#8B5CF6', width=3, dash='dash'), marker=dict(size=8, color='#EC4899'))
    
    fig_elbow.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

# -------------------------------------------------------------
# 9. Clean Data Preview Area
# -------------------------------------------------------------
st.markdown("---")
with st.expander("🔍 Inspect Processed Dataset Preview", expanded=False):
    st.markdown("The interactive table below showcases the values parsed directly into the machine learning algorithm based on your sidebar specifications.")
    st.dataframe(
        df_cluster.head(15).style.format(precision=3).background_gradient(cmap='Blues', subset=['petal length (cm)', 'petal width (cm)']), 
        use_container_width=True
    )
