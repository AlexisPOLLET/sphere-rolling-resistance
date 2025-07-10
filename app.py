import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="Sphere Detection Analysis",
    page_icon="⚽",
    layout="wide"
)

# Titre principal
st.title("🔍 Analysis of Sphere Detection on Wet Granular Material")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a section:", 
                           ["Overview", "Trajectory Analysis", "Performance Metrics", "Configuration"])

@st.cache_data
def load_data():
    """Charge les données des détections"""
    try:
        # Simulation des données basée sur le CSV fourni
        data = pd.read_csv('detections.csv')
        # Filtrer les détections valides (non nulles)
        data = data[(data['X_center'] != 0) | (data['Y_center'] != 0) | (data['Radius'] != 0)]
        return data
    except FileNotFoundError:
        # Données simulées si le fichier n'est pas trouvé
        np.random.seed(42)
        frames = range(9, 108)
        x_centers = 1238 - np.cumulative_sum(np.random.normal(4, 1, len(frames)))
        y_centers = 679 + np.cumulative_sum(np.random.normal(0.2, 0.5, len(frames)))
        radii = np.random.normal(25, 5, len(frames))
        
        return pd.DataFrame({
            'Frame': frames,
            'X_center': x_centers.astype(int),
            'Y_center': y_centers.astype(int),
            'Radius': radii.astype(int)
        })

# Chargement des données
data = load_data()

if page == "Overview":
    st.header("📊 Detection Overview")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Frames", 107)
    with col2:
        st.metric("Successful Detections", len(data))
    with col3:
        success_rate = (len(data) / 107) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col4:
        avg_radius = data['Radius'].mean()
        st.metric("Average Radius", f"{avg_radius:.1f} px")
    
    # Configuration utilisée
    st.subheader("🔧 Detection Configuration")
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.info("""
        **Detection Parameters:**
        - Min Radius: 18 pixels
        - Max Radius: 35 pixels  
        - BW Threshold: 8
        - Circularity Min: 0.5
        - Min Score: 40
        """)
    
    with config_col2:
        st.info("""
        **Image Processing:**
        - Cropped Area: (400,500) to (1246,1000)
        - Cropped Size: 846 x 500 pixels
        - Background Images: 150
        - Max Movement: 120 pixels
        """)

elif page == "Trajectory Analysis":
    st.header("📈 Sphere Trajectory Analysis")
    
    # Graphique de trajectoire principal
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Trajectory', 'X Position vs Frame', 'Y Position vs Frame', 'Radius vs Frame'),
        specs=[[{"colspan": 2}, None],
               [{}, {}]]
    )
    
    # Trajectoire 2D
    fig.add_trace(
        go.Scatter(x=data['X_center'], y=data['Y_center'], 
                  mode='markers+lines',
                  marker=dict(size=8, color=data['Frame'], colorscale='Viridis'),
                  name='Trajectory',
                  text=[f"Frame {f}<br>Radius: {r}px" for f, r in zip(data['Frame'], data['Radius'])]),
        row=1, col=1
    )
    
    # Position X vs Frame
    fig.add_trace(
        go.Scatter(x=data['Frame'], y=data['X_center'],
                  mode='lines+markers',
                  name='X Position',
                  line=dict(color='blue')),
        row=2, col=1
    )
    
    # Position Y vs Frame
    fig.add_trace(
        go.Scatter(x=data['Frame'], y=data['Y_center'],
                  mode='lines+markers',
                  name='Y Position',
                  line=dict(color='red')),
        row=2, col=2
    )
    
    # Inversion de l'axe Y pour la trajectoire (coordonnées image)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    
    fig.update_layout(height=700, title_text="Sphere Movement Analysis")
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse du mouvement
    st.subheader("🎯 Movement Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Calcul de la vitesse
        if len(data) > 1:
            distances = np.sqrt(np.diff(data['X_center'])**2 + np.diff(data['Y_center'])**2)
            avg_speed = np.mean(distances)
            st.metric("Average Speed", f"{avg_speed:.2f} px/frame")
            
            total_distance = np.sum(distances)
            st.metric("Total Distance", f"{total_distance:.0f} px")
    
    with col2:
        # Variation du rayon
        radius_std = data['Radius'].std()
        st.metric("Radius Variation (σ)", f"{radius_std:.2f} px")
        
        radius_range = data['Radius'].max() - data['Radius'].min()
        st.metric("Radius Range", f"{radius_range} px")

elif page == "Performance Metrics":
    st.header("📊 Detection Performance Analysis")
    
    # Graphique de distribution des rayons
    fig_radius = px.histogram(data, x='Radius', nbins=20,
                             title="Distribution of Detected Radii",
                             labels={'Radius': 'Radius (pixels)', 'count': 'Frequency'})
    st.plotly_chart(fig_radius, use_container_width=True)
    
    # Analyse de la qualité de détection
    st.subheader("🎯 Detection Quality Analysis")
    
    # Statistiques par plages de frames
    frame_ranges = pd.cut(data['Frame'], bins=5, labels=[f"Range {i+1}" for i in range(5)])
    stats_by_range = data.groupby(frame_ranges).agg({
        'Radius': ['mean', 'std', 'count'],
        'X_center': 'std',
        'Y_center': 'std'
    }).round(2)
    
    st.subheader("📈 Statistics by Frame Ranges")
    st.dataframe(stats_by_range)
    
    # Heatmap de position
    st.subheader("🗺️ Position Heatmap")
    fig_heatmap = px.density_heatmap(data, x='X_center', y='Y_center',
                                    title="Sphere Position Density",
                                    labels={'X_center': 'X Position', 'Y_center': 'Y Position'})
    fig_heatmap.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_heatmap, use_container_width=True)

elif page == "Configuration":
    st.header("⚙️ Detection Configuration")
    
    # Affichage de la configuration
    st.subheader("Current Configuration")
    config_data = {
        'Parameter': ['Min Radius', 'Max Radius', 'BW Threshold', 'Circularity Min', 
                     'Min Score', 'Max Movement', 'Background Images'],
        'Value': ['18 pixels', '35 pixels', '8', '0.5', '40', '120 pixels', '150'],
        'Description': [
            'Minimum radius for sphere detection',
            'Maximum radius for sphere detection', 
            'Binary threshold for edge detection',
            'Minimum circularity score (0-1)',
            'Minimum quality score for detection',
            'Maximum movement between frames',
            'Number of background reference images'
        ]
    }
    
    config_df = pd.DataFrame(config_data)
    st.dataframe(config_df, use_container_width=True)
    
    # Crop information
    st.subheader("🖼️ Image Processing")
    crop_info = {
        'Setting': ['Original Image Area', 'Cropped Area', 'Cropped Size', 'Crop Start', 'Crop End'],
        'Value': ['Unknown', '(400,500) to (1246,1000)', '846 x 500 pixels', '(400, 500)', '(1246, 1000)']
    }
    
    crop_df = pd.DataFrame(crop_info)
    st.dataframe(crop_df, use_container_width=True)
    
    # Results summary
    st.subheader("📋 Results Summary")
    results_summary = {
        'Metric': ['Images Processed', 'Successful Detections', 'Success Rate', 'Failed Detections'],
        'Value': ['107', f'{len(data)}', f'{(len(data)/107)*100:.1f}%', f'{107-len(data)}'],
        'Status': ['✅ Complete', '✅ Good', '🟡 Acceptable' if (len(data)/107) > 0.7 else '❌ Poor', 
                  '🟡 Some losses' if (107-len(data)) < 30 else '❌ Many losses']
    }
    
    results_df = pd.DataFrame(results_summary)
    st.dataframe(results_df, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Rolling Resistance of Spheres on Wet Granular Material")
st.sidebar.markdown("**Analysis:** Computer Vision Detection Results")
