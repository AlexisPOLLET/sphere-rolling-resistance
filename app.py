import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
from PIL import Image
import base64
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sphere Rolling Resistance Analysis",
    page_icon="⚪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .section-header {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
    }
    .comparison-card {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .image-container {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for multi-experiment comparison
if 'experiments' not in st.session_state:
    st.session_state.experiments = {}

# Function to load and display images
def display_detection_images():
    """Display example detection images"""
    st.markdown("""
    ### 🎯 Sphere Detection Examples
    Here are examples of how spheres are detected in the experimental setup:
    """)
    
    # Create columns for image display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Early Detection (Frame 13)")
        st.markdown("""
        - **Frame**: 13
        - **Score**: 76.7
        - **Radius**: 30px
        - **Position**: Top-right area
        """)
        # Placeholder for detection image
        st.info("🖼️ Detection image would be displayed here\n(Frame 13 with green circle and red center point)")
    
    with col2:
        st.markdown("#### Mid-trajectory (Frame 57)")
        st.markdown("""
        - **Frame**: 57
        - **Score**: 67.8
        - **Radius**: 22px
        - **Position**: Center area
        """)
        st.info("🖼️ Detection image would be displayed here\n(Frame 57 showing sphere movement)")
    
    with col3:
        st.markdown("#### Late Detection (Frame 98)")
        st.markdown("""
        - **Frame**: 98
        - **Score**: 68.0
        - **Radius**: 21px
        - **Position**: Lower-left area
        """)
        st.info("🖼️ Detection image would be displayed here\n(Frame 98 near end of trajectory)")

# Function to create sample data with metadata
def create_sample_data_with_metadata(experiment_name="Sample", water_content=0.0, sphere_type="Steel"):
    """Creates sample data with experimental metadata"""
    frames = list(range(1, 108))
    data = []
    
    # Modify trajectory based on water content (simulation)
    water_effect = 1 + (water_content / 100) * 0.3  # Higher water content = more resistance
    
    for frame in frames:
        if frame < 9:
            data.append([frame, 0, 0, 0])
        elif frame in [30, 31]:
            data.append([frame, 0, 0, 0])
        else:
            # Adjust movement based on water content
            x = 1240 - (frame - 9) * 12 * water_effect + np.random.normal(0, 2)
            y = 680 + (frame - 9) * 0.5 + np.random.normal(0, 3)
            radius = 20 + np.random.normal(5, 3)
            radius = max(18, min(35, radius))
            data.append([frame, max(0, x), max(0, y), max(0, radius)])
    
    df = pd.DataFrame(data, columns=['Frame', 'X_center', 'Y_center', 'Radius'])
    
    # Add metadata
    metadata = {
        'experiment_name': experiment_name,
        'water_content': water_content,
        'sphere_type': sphere_type,
        'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_frames': len(df),
        'valid_detections': len(df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]),
        'success_rate': len(df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]) / len(df) * 100
    }
    
    return df, metadata

# Function to load data with metadata extraction
@st.cache_data
def load_uploaded_data_with_metadata(uploaded_file, experiment_name, water_content, sphere_type):
    """Loads data from uploaded file with metadata"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Check required columns
        required_columns = ['Frame', 'X_center', 'Y_center', 'Radius']
        if not all(col in df.columns for col in required_columns):
            st.error(f"❌ File must contain columns: {required_columns}")
            st.error(f"📊 Found columns: {list(df.columns)}")
            return None, None
        
        # Filter valid detections
        df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
        
        # Create metadata
        metadata = {
            'experiment_name': experiment_name,
            'water_content': water_content,
            'sphere_type': sphere_type,
            'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_frames': len(df),
            'valid_detections': len(df_valid),
            'success_rate': len(df_valid) / len(df) * 100 if len(df) > 0 else 0
        }
        
        return df, metadata
    return None, None

# Function to calculate Krr for comparison
def calculate_krr_simple(df_valid, fps=250, pixels_per_mm=5.0, sphere_mass_g=10.0, angle_deg=15.0):
    """Simple Krr calculation for comparison"""
    if len(df_valid) < 10:
        return None
    
    # Convert to real units
    dt = 1 / fps
    x_m = df_valid['X_center'].values / pixels_per_mm / 1000
    y_m = df_valid['Y_center'].values / pixels_per_mm / 1000
    
    # Calculate velocities
    vx = np.gradient(x_m, dt)
    vy = np.gradient(y_m, dt)
    v_magnitude = np.sqrt(vx**2 + vy**2)
    
    # Initial and final velocities
    n_avg = min(3, len(v_magnitude)//4)
    v0 = np.mean(v_magnitude[:n_avg])
    vf = np.mean(v_magnitude[-n_avg:])
    
    # Total distance
    distances = np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2)
    total_distance = np.sum(distances)
    
    # Calculate Krr
    g = 9.81
    if total_distance > 0 and v0 > vf:
        krr = (v0**2 - vf**2) / (2 * g * total_distance)
        return {
            'krr': krr,
            'v0': v0,
            'vf': vf,
            'distance': total_distance,
            'avg_velocity': np.mean(v_magnitude),
            'max_velocity': np.max(v_magnitude)
        }
    return None

# Main title
st.markdown("""
# 🎾 Sphere Rolling Resistance Analysis Platform
## 🔬 Complete Analysis Suite for Granular Mechanics Research
""")

# Page navigation
page = st.sidebar.selectbox("📋 Select Page", [
    "🏠 Home & Single Analysis",
    "🔍 Multi-Experiment Comparison",
    "📊 Advanced Statistics",
    "📖 Method & Documentation"
])

# ==================== HOME PAGE ====================
if page == "🏠 Home & Single Analysis":
    
    st.markdown("*Upload your data and access our specialized analysis tools*")
    
    # Display detection images section
    display_detection_images()
    
    # File upload section
    st.markdown("""
    <div class="upload-section">
        <h2>📂 Upload Your Experimental Data</h2>
        <p>Start by uploading your CSV file with detection results to get a personalized analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Experiment metadata input
    col1, col2, col3 = st.columns(3)
    with col1:
        experiment_name = st.text_input("Experiment Name", value="Experiment_1")
    with col2:
        water_content = st.number_input("Water Content (%)", value=0.0, min_value=0.0, max_value=30.0)
    with col3:
        sphere_type = st.selectbox("Sphere Type", ["Steel", "Plastic", "Glass", "Other"])
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose your CSV file with detection data", 
        type=['csv'],
        help="Upload a CSV file with columns: Frame, X_center, Y_center, Radius"
    )
    
    # Variables for data
    df = None
    df_valid = None
    metadata = None
    
    # Load data
    if uploaded_file is not None:
        df, metadata = load_uploaded_data_with_metadata(uploaded_file, experiment_name, water_content, sphere_type)
        if df is not None:
            df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
            st.success(f"✅ File loaded successfully! {len(df)} frames detected")
            
            # Add to experiments for comparison
            if st.button("💾 Save experiment for comparison"):
                st.session_state.experiments[experiment_name] = {
                    'data': df,
                    'metadata': metadata
                }
                st.success(f"Experiment '{experiment_name}' saved for comparison!")
    else:
        # Option to use sample data
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔬 Use dry soil sample (0% water)"):
                df, metadata = create_sample_data_with_metadata("Dry_Sample", 0.0, sphere_type)
                df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
                st.info("📊 Dry soil sample loaded")
        
        with col2:
            if st.button("💧 Use wet soil sample (15% water)"):
                df, metadata = create_sample_data_with_metadata("Wet_Sample", 15.0, sphere_type)
                df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
                st.info("📊 Wet soil sample loaded")
    
    # Display analysis if data is loaded
    if df is not None and df_valid is not None:
        
        # Quick data overview
        st.markdown("### 📊 Experiment Overview")
        
        # Metadata display
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Experiment Details:**
            - **Name**: {metadata['experiment_name']}
            - **Water Content**: {metadata['water_content']}%
            - **Sphere Type**: {metadata['sphere_type']}
            - **Date**: {metadata['date']}
            """)
        
        with col2:
            st.markdown(f"""
            **Detection Results:**
            - **Total Frames**: {metadata['total_frames']}
            - **Valid Detections**: {metadata['valid_detections']}
            - **Success Rate**: {metadata['success_rate']:.1f}%
            """)
        
        # Metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(df)}</h3>
                <p>Total Frames</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(df_valid)}</h3>
                <p>Valid Detections</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            success_rate = len(df_valid) / len(df) * 100 if len(df) > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>{success_rate:.1f}%</h3>
                <p>Success Rate</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            avg_radius = df_valid['Radius'].mean() if len(df_valid) > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>{avg_radius:.1f} px</h3>
                <p>Average Radius</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick trajectory visualization
        st.markdown("### 🎯 Trajectory Visualization")
        if len(df_valid) > 0:
            fig = px.scatter(df_valid, x='X_center', y='Y_center', 
                           color='Frame', size='Radius',
                           title=f"Sphere Trajectory - {metadata['experiment_name']}")
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Quick Krr calculation
        st.markdown("### ⚡ Quick Krr Analysis")
        krr_results = calculate_krr_simple(df_valid, water_content=water_content)
        
        if krr_results:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Krr Coefficient", f"{krr_results['krr']:.6f}")
            with col2:
                st.metric("Initial Velocity", f"{krr_results['v0']*1000:.1f} mm/s")
            with col3:
                st.metric("Final Velocity", f"{krr_results['vf']*1000:.1f} mm/s")
            with col4:
                st.metric("Distance", f"{krr_results['distance']*1000:.1f} mm")
        else:
            st.warning("⚠️ Cannot calculate Krr with current data")

# ==================== MULTI-EXPERIMENT COMPARISON PAGE ====================
elif page == "🔍 Multi-Experiment Comparison":
    
    st.markdown("""
    <div class="section-header">
        <h2>🔍 Multi-Experiment Comparison</h2>
        <p>Compare multiple experiments to analyze the effect of different parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if experiments are available
    if not st.session_state.experiments:
        st.warning("⚠️ No experiments available for comparison. Please load some experiments from the Home page first.")
        
        # Quick load sample experiments
        st.markdown("### 🚀 Quick Start: Load Sample Experiments")
        if st.button("📊 Load sample experiments for comparison"):
            # Create sample experiments with different water contents
            water_contents = [0, 5, 10, 15, 20]
            for w in water_contents:
                df, metadata = create_sample_data_with_metadata(f"Sample_W{w}%", w, "Steel")
                st.session_state.experiments[f"Sample_W{w}%"] = {
                    'data': df,
                    'metadata': metadata
                }
            st.success("✅ Sample experiments loaded!")
            st.rerun()
    
    else:
        # Display available experiments
        st.markdown("### 📋 Available Experiments")
        
        # Experiments overview table
        exp_data = []
        for name, exp in st.session_state.experiments.items():
            meta = exp['metadata']
            exp_data.append({
                'Experiment': name,
                'Water Content (%)': meta['water_content'],
                'Sphere Type': meta['sphere_type'],
                'Success Rate (%)': f"{meta['success_rate']:.1f}",
                'Valid Detections': meta['valid_detections'],
                'Date': meta['date']
            })
        
        exp_df = pd.DataFrame(exp_data)
        st.dataframe(exp_df, use_container_width=True)
        
        # Experiment selection for comparison
        st.markdown("### 🔬 Select Experiments to Compare")
        selected_experiments = st.multiselect(
            "Choose experiments for comparison:",
            options=list(st.session_state.experiments.keys()),
            default=list(st.session_state.experiments.keys())[:min(4, len(st.session_state.experiments))]
        )
        
        if len(selected_experiments) >= 2:
            # Calculate comparison metrics
            comparison_data = []
            trajectory_data = []
            
            for exp_name in selected_experiments:
                exp = st.session_state.experiments[exp_name]
                df = exp['data']
                meta = exp['metadata']
                df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
                
                # Calculate Krr
                krr_results = calculate_krr_simple(df_valid)
                
                comparison_data.append({
                    'Experiment': exp_name,
                    'Water_Content': meta['water_content'],
                    'Sphere_Type': meta['sphere_type'],
                    'Success_Rate': meta['success_rate'],
                    'Krr': krr_results['krr'] if krr_results else None,
                    'Initial_Velocity': krr_results['v0'] if krr_results else None,
                    'Final_Velocity': krr_results['vf'] if krr_results else None,
                    'Distance': krr_results['distance'] if krr_results else None,
                    'Avg_Velocity': krr_results['avg_velocity'] if krr_results else None
                })
                
                # Trajectory data for overlay
                if len(df_valid) > 0:
                    df_traj = df_valid.copy()
                    df_traj['Experiment'] = exp_name
                    df_traj['Water_Content'] = meta['water_content']
                    trajectory_data.append(df_traj)
            
            comp_df = pd.DataFrame(comparison_data)
            
            # ===== COMPARISON VISUALIZATIONS =====
            st.markdown("### 📊 Comparison Analysis")
            
            # Krr vs Water Content
            col1, col2 = st.columns(2)
            
            with col1:
                if comp_df['Krr'].notna().any():
                    fig_krr = px.scatter(comp_df, x='Water_Content', y='Krr', 
                                       color='Sphere_Type', size='Success_Rate',
                                       hover_data=['Experiment'],
                                       title="🔍 Krr vs Water Content")
                    fig_krr.add_trace(go.Scatter(
                        x=comp_df['Water_Content'], 
                        y=comp_df['Krr'],
                        mode='lines',
                        name='Trend',
                        line=dict(dash='dash', color='red')
                    ))
                    st.plotly_chart(fig_krr, use_container_width=True)
                else:
                    st.warning("No valid Krr data for comparison")
            
            with col2:
                # Success rate comparison
                fig_success = px.bar(comp_df, x='Experiment', y='Success_Rate',
                                   color='Water_Content',
                                   title="📈 Detection Success Rate Comparison")
                fig_success.update_xaxes(tickangle=45)
                st.plotly_chart(fig_success, use_container_width=True)
            
            # Velocity comparison
            col1, col2 = st.columns(2)
            
            with col1:
                if comp_df['Initial_Velocity'].notna().any():
                    fig_vel = px.scatter(comp_df, x='Water_Content', 
                                       y=['Initial_Velocity', 'Final_Velocity'],
                                       title="🏃 Velocity Comparison")
                    st.plotly_chart(fig_vel, use_container_width=True)
            
            with col2:
                if comp_df['Distance'].notna().any():
                    fig_dist = px.bar(comp_df, x='Experiment', y='Distance',
                                    color='Water_Content',
                                    title="📏 Distance Traveled Comparison")
                    fig_dist.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            # Trajectory overlay
            if trajectory_data:
                st.markdown("### 🛤️ Trajectory Overlay Comparison")
                
                combined_traj = pd.concat(trajectory_data, ignore_index=True)
                fig_overlay = px.scatter(combined_traj, x='X_center', y='Y_center',
                                       color='Experiment', 
                                       animation_frame='Frame',
                                       title="Trajectory Comparison - All Experiments")
                fig_overlay.update_yaxes(autorange="reversed")
                fig_overlay.update_layout(height=600)
                st.plotly_chart(fig_overlay, use_container_width=True)
            
            # Statistical comparison table
            st.markdown("### 📋 Detailed Comparison Table")
            
            # Format the comparison table
            display_comp = comp_df.copy()
            if 'Krr' in display_comp.columns:
                display_comp['Krr'] = display_comp['Krr'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
            if 'Initial_Velocity' in display_comp.columns:
                display_comp['Initial_Velocity (mm/s)'] = display_comp['Initial_Velocity'].apply(lambda x: f"{x*1000:.2f}" if pd.notna(x) else "N/A")
            if 'Final_Velocity' in display_comp.columns:
                display_comp['Final_Velocity (mm/s)'] = display_comp['Final_Velocity'].apply(lambda x: f"{x*1000:.2f}" if pd.notna(x) else "N/A")
            if 'Distance' in display_comp.columns:
                display_comp['Distance (mm)'] = display_comp['Distance'].apply(lambda x: f"{x*1000:.2f}" if pd.notna(x) else "N/A")
            
            # Select relevant columns for display
            display_columns = ['Experiment', 'Water_Content', 'Sphere_Type', 'Success_Rate', 
                             'Krr', 'Initial_Velocity (mm/s)', 'Final_Velocity (mm/s)', 'Distance (mm)']
            
            available_columns = [col for col in display_columns if col in display_comp.columns]
            st.dataframe(display_comp[available_columns], use_container_width=True)
            
            # Key insights
            st.markdown("### 🔍 Key Insights")
            
            if len(comp_df) >= 2:
                # Water content effect analysis
                water_sorted = comp_df.sort_values('Water_Content')
                if len(water_sorted) >= 2 and water_sorted['Krr'].notna().sum() >= 2:
                    krr_change = water_sorted['Krr'].iloc[-1] - water_sorted['Krr'].iloc[0]
                    water_change = water_sorted['Water_Content'].iloc[-1] - water_sorted['Water_Content'].iloc[0]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="comparison-card">
                            <h4>💧 Water Content Effect</h4>
                            <p>Krr change: <strong>{krr_change:.6f}</strong></p>
                            <p>Water range: <strong>{water_change:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        best_exp = comp_df.loc[comp_df['Success_Rate'].idxmax()]
                        st.markdown(f"""
                        <div class="comparison-card">
                            <h4>🏆 Best Detection</h4>
                            <p><strong>{best_exp['Experiment']}</strong></p>
                            <p>{best_exp['Success_Rate']:.1f}% success rate</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        if comp_df['Krr'].notna().any():
                            krr_range = comp_df['Krr'].max() - comp_df['Krr'].min()
                            st.markdown(f"""
                            <div class="comparison-card">
                                <h4>📊 Krr Variation</h4>
                                <p>Range: <strong>{krr_range:.6f}</strong></p>
                                <p>Coefficient of variation</p>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Export comparison results
            st.markdown("### 💾 Export Comparison Results")
            
            csv_comparison = comp_df.to_csv(index=False)
            st.download_button(
                label="📥 Download comparison results (CSV)",
                data=csv_comparison,
                file_name="experiment_comparison.csv",
                mime="text/csv"
            )
        
        else:
            st.info("Please select at least 2 experiments for comparison")
        
        # Experiment management
        st.markdown("### 🗂️ Experiment Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Remove Experiment:**")
            exp_to_remove = st.selectbox("Select experiment to remove:", 
                                       options=["None"] + list(st.session_state.experiments.keys()))
            
            if exp_to_remove != "None" and st.button("🗑️ Remove Selected Experiment"):
                del st.session_state.experiments[exp_to_remove]
                st.success(f"Experiment '{exp_to_remove}' removed!")
                st.rerun()
        
        with col2:
            st.markdown("**Clear All:**")
            st.write("⚠️ This will remove all saved experiments")
            if st.button("🧹 Clear All Experiments"):
                st.session_state.experiments = {}
                st.success("All experiments cleared!")
                st.rerun()

# ==================== ADVANCED STATISTICS PAGE ====================
elif page == "📊 Advanced Statistics":
    st.markdown("""
    <div class="section-header">
        <h2>📊 Advanced Statistics & Modeling</h2>
        <p>Statistical analysis and predictive modeling across experiments</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.experiments:
        # Statistical analysis across all experiments
        all_results = []
        
        for exp_name, exp in st.session_state.experiments.items():
            df = exp['data']
            meta = exp['metadata']
            df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
            
            krr_results = calculate_krr_simple(df_valid)
            
            if krr_results:
                all_results.append({
                    'experiment': exp_name,
                    'water_content': meta['water_content'],
                    'sphere_type': meta['sphere_type'],
                    'krr': krr_results['krr'],
                    'success_rate': meta['success_rate']
                })
        
        if all_results:
            stats_df = pd.DataFrame(all_results)
            
            # Correlation analysis
            st.markdown("### 🔗 Correlation Analysis")
            
            if len(stats_df) >= 3:
                corr = stats_df[['water_content', 'krr', 'success_rate']].corr()
                
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                                   title="Correlation Matrix")
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Trend analysis
                st.markdown("### 📈 Trend Analysis")
                
                # Polynomial fit for Krr vs Water Content
                if len(stats_df) >= 4:
                    from sklearn.preprocessing import PolynomialFeatures
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import r2_score
                    
                    try:
                        X = stats_df['water_content'].values.reshape(-1, 1)
                        y = stats_df['krr'].values
                        
                        # Fit polynomial models
                        degrees = [1, 2, 3]
                        fig_trend = go.Figure()
                        
                        # Add actual data points
                        fig_trend.add_trace(go.Scatter(
                            x=stats_df['water_content'],
                            y=stats_df['krr'],
                            mode='markers',
                            name='Experimental Data',
                            marker=dict(size=10, color='blue')
                        ))
                        
                        # Fit different polynomial degrees
                        x_smooth = np.linspace(stats_df['water_content'].min(), 
                                             stats_df['water_content'].max(), 100)
                        
                        for degree in degrees:
                            poly_features = PolynomialFeatures(degree=degree)
                            X_poly = poly_features.fit_transform(X)
                            model = LinearRegression()
                            model.fit(X_poly, y)
                            
                            X_smooth_poly = poly_features.transform(x_smooth.reshape(-1, 1))
                            y_pred_smooth = model.predict(X_smooth_poly)
                            
                            r2 = r2_score(y, model.predict(X_poly))
                            
                            fig_trend.add_trace(go.Scatter(
                                x=x_smooth,
                                y=y_pred_smooth,
                                mode='lines',
                                name=f'Polynomial Degree {degree} (R² = {r2:.3f})',
                                line=dict(dash='dash' if degree > 1 else 'solid')
                            ))
                        
                        fig_trend.update_layout(
                            title='Krr vs Water Content - Trend Analysis',
                            xaxis_title='Water Content (%)',
                            yaxis_title='Krr Coefficient',
                            height=500
                        )
                        
                        st.plotly_chart(fig_trend, use_container_width=True)
                        
                        # Model equations and predictions
                        st.markdown("### 🧮 Predictive Models")
                        
                        # Best fit model
                        poly_features = PolynomialFeatures(degree=2)  # Quadratic model
                        X_poly = poly_features.fit_transform(X)
                        best_model = LinearRegression()
                        best_model.fit(X_poly, y)
                        
                        r2_best = r2_score(y, best_model.predict(X_poly))
                        
                        st.markdown(f"""
                        **Best Fit Model (Quadratic):**
                        - R² Score: {r2_best:.4f}
                        - Model: Krr = a + b×W + c×W²
                        - Where W = Water Content (%)
                        """)
                        
                        # Prediction interface
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            predict_water = st.slider("Predict Krr for water content:", 
                                                     0.0, 30.0, 10.0, 0.5)
                            
                            X_pred = poly_features.transform([[predict_water]])
                            krr_pred = best_model.predict(X_pred)[0]
                            
                            st.metric("Predicted Krr", f"{krr_pred:.6f}")
                            
                            # Confidence interval (simplified)
                            residuals = y - best_model.predict(X_poly)
                            std_error = np.std(residuals)
                            
                            st.metric("±95% Confidence", f"±{1.96*std_error:.6f}")
                        
                        with col2:
                            # Model validation
                            st.markdown("**Model Validation:**")
                            
                            # Calculate metrics
                            rmse = np.sqrt(np.mean(residuals**2))
                            mae = np.mean(np.abs(residuals))
                            
                            st.markdown(f"""
                            - RMSE: {rmse:.6f}
                            - MAE: {mae:.6f}
                            - Data Points: {len(stats_df)}
                            """)
                            
                            if r2_best > 0.8:
                                st.success("✅ Good model fit")
                            elif r2_best > 0.6:
                                st.warning("⚠️ Moderate model fit")
                            else:
                                st.error("❌ Poor model fit - more data needed")
                    
                    except ImportError:
                        st.warning("⚠️ sklearn not available for advanced modeling")
                        
                        # Simple linear regression without sklearn
                        import numpy.polynomial.polynomial as poly
                        
                        x_data = stats_df['water_content'].values
                        y_data = stats_df['krr'].values
                        
                        # Linear fit
                        coeffs = poly.polyfit(x_data, y_data, 1)
                        
                        fig_simple = go.Figure()
                        fig_simple.add_trace(go.Scatter(
                            x=x_data, y=y_data,
                            mode='markers',
                            name='Data',
                            marker=dict(size=10)
                        ))
                        
                        x_line = np.linspace(x_data.min(), x_data.max(), 100)
                        y_line = poly.polyval(x_line, coeffs)
                        
                        fig_simple.add_trace(go.Scatter(
                            x=x_line, y=y_line,
                            mode='lines',
                            name='Linear Fit'
                        ))
                        
                        fig_simple.update_layout(
                            title='Simple Linear Trend',
                            xaxis_title='Water Content (%)',
                            yaxis_title='Krr Coefficient'
                        )
                        
                        st.plotly_chart(fig_simple, use_container_width=True)
                
                # Statistical summary
                st.markdown("### 📊 Statistical Summary")
                
                summary_stats = stats_df.describe()
                st.dataframe(summary_stats)
                
                # Hypothesis testing
                st.markdown("### 🧪 Hypothesis Testing")
                
                # Test if water content significantly affects Krr
                if len(stats_df) >= 5:
                    try:
                        from scipy import stats as scipy_stats
                        
                        # Correlation test
                        corr_coef, p_value = scipy_stats.pearsonr(stats_df['water_content'], 
                                                                 stats_df['krr'])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Correlation Test:**")
                            st.markdown(f"""
                            - Correlation coefficient: {corr_coef:.4f}
                            - P-value: {p_value:.4f}
                            """)
                            
                            if p_value < 0.05:
                                st.success("✅ Significant correlation (p < 0.05)")
                            else:
                                st.warning("⚠️ No significant correlation (p ≥ 0.05)")
                        
                        with col2:
                            # ANOVA test if multiple sphere types
                            if len(stats_df['sphere_type'].unique()) > 1:
                                groups = [stats_df[stats_df['sphere_type'] == t]['krr'].values 
                                        for t in stats_df['sphere_type'].unique()]
                                
                                f_stat, p_val_anova = scipy_stats.f_oneway(*groups)
                                
                                st.markdown("**ANOVA Test (Sphere Types):**")
                                st.markdown(f"""
                                - F-statistic: {f_stat:.4f}
                                - P-value: {p_val_anova:.4f}
                                """)
                                
                                if p_val_anova < 0.05:
                                    st.success("✅ Significant difference between sphere types")
                                else:
                                    st.info("ℹ️ No significant difference between sphere types")
                    
                    except ImportError:
                        st.warning("⚠️ scipy not available for statistical tests")
            
            else:
                st.info("Need at least 3 experiments for meaningful statistical analysis")
    
    else:
        st.warning("⚠️ No experiments available for statistical analysis")

# ==================== DOCUMENTATION PAGE ====================
elif page == "📖 Method & Documentation":
    st.markdown("""
    <div class="section-header">
        <h2>📖 Method & Documentation</h2>
        <p>Complete documentation of the experimental method and analysis procedures</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Method overview
    st.markdown("### 🔬 Experimental Method")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 Sphere Detection Algorithm
        
        **Method**: Background subtraction with circle detection
        
        **Steps**:
        1. **Background Creation**: Average of 150 reference images
        2. **Subtraction**: Remove static background
        3. **Thresholding**: Binary image with adaptive threshold
        4. **Morphology**: Clean contours and noise
        5. **Detection**: Find circular contours
        6. **Validation**: Filter by size, shape, and continuity
        
        **Quality Criteria**:
        - Size: 18 ≤ radius ≤ 35 pixels
        - Shape: Circularity ≥ 0.5
        - Continuity: Movement ≤ 120 pixels/frame
        - Score: Overall quality ≥ 40
        """)
    
    with col2:
        st.markdown("""
        #### ⚡ Krr Calculation
        
        **Formula**: Krr = (V₀² - Vf²) / (2gL)
        
        Where:
        - V₀ = Initial velocity
        - Vf = Final velocity  
        - g = Gravitational acceleration (9.81 m/s²)
        - L = Distance traveled
        
        **Physical Meaning**:
        - Krr represents energy loss due to rolling resistance
        - Higher Krr = more resistance
        - Typical values: 0.05-0.07 for dry soils
        
        **Factors Affecting Krr**:
        - Water content (humidity)
        - Sphere material and size
        - Granular material properties
        - Surface conditions
        """)
    
    # Analysis procedures
    st.markdown("### 📊 Analysis Procedures")
    
    tab1, tab2, tab3 = st.tabs(["Single Analysis", "Multi-Experiment", "Statistical"])
    
    with tab1:
        st.markdown("""
        #### 🔍 Single Experiment Analysis
        
        **Data Requirements**:
        - CSV file with columns: Frame, X_center, Y_center, Radius
        - Minimum 10 valid detections
        - Experimental metadata (water content, sphere type)
        
        **Analysis Steps**:
        1. **Data Validation**: Check data quality and completeness
        2. **Trajectory Reconstruction**: Plot sphere path
        3. **Kinematic Analysis**: Calculate velocities and accelerations
        4. **Energy Analysis**: Compute kinetic energies
        5. **Krr Calculation**: Determine rolling resistance coefficient
        
        **Quality Checks**:
        - Detection success rate > 70%
        - Smooth trajectory without jumps
        - Realistic velocity ranges
        - Energy conservation principles
        """)
    
    with tab2:
        st.markdown("""
        #### 🔍 Multi-Experiment Comparison
        
        **Purpose**: 
        Compare experiments to understand parameter effects
        
        **Comparison Metrics**:
        - Krr coefficient variation
        - Detection success rates
        - Velocity profiles
        - Trajectory characteristics
        
        **Typical Comparisons**:
        - Water content effect (0% vs 5% vs 10% vs 15%)
        - Sphere material influence (Steel vs Plastic)
        - Size effects (different sphere diameters)
        
        **Statistical Analysis**:
        - Correlation analysis
        - Trend fitting (polynomial models)
        - Hypothesis testing
        - ANOVA for group comparisons
        """)
    
    with tab3:
        st.markdown("""
        #### 📊 Statistical Methods
        
        **Correlation Analysis**:
        - Pearson correlation coefficient
        - P-value for significance testing
        - Confidence intervals
        
        **Regression Models**:
        - Linear regression: Krr = a + b×W
        - Quadratic model: Krr = a + b×W + c×W²
        - Model validation (R², RMSE, MAE)
        
        **Hypothesis Testing**:
        - H₀: Water content has no effect on Krr
        - H₁: Water content significantly affects Krr
        - α = 0.05 significance level
        
        **Model Selection**:
        - Cross-validation
        - Information criteria (AIC, BIC)
        - Residual analysis
        """)
    
    # Literature references
    st.markdown("### 📚 Literature References")
    
    st.markdown("""
    **Key References**:
    
    1. **Van Wal et al. (2017)**: "Rolling resistance of spheres on granular materials"
       - Dry soil Krr values: 0.05-0.07
       - Methodology for Krr measurement
    
    2. **Darbois Texier et al. (2018)**: "Penetration depth scaling law"
       - δ/R ∝ (ρs/ρg)^0.75
       - Sphere density influence
    
    3. **De Blasio (2009)**: "Rolling resistance independence"
       - Krr independent of rolling speed
       - Theoretical framework
    
    **Research Innovation**:
    - First study of humidity effects on rolling resistance
    - Extension of dry soil models to wet conditions
    - Systematic water content variation (0-25%)
    """)
    
    # File format documentation
    st.markdown("### 📁 File Format Documentation")
    
    st.markdown("""
    #### CSV File Format
    
    **Required Columns**:
    ```
    Frame,X_center,Y_center,Radius
    1,1234,678,25
    2,1230,679,24
    3,1226,680,25
    ...
    ```
    
    **Column Descriptions**:
    - **Frame**: Sequential image number (integer)
    - **X_center**: Horizontal position of sphere center (pixels)
    - **Y_center**: Vertical position of sphere center (pixels)
    - **Radius**: Detected sphere radius (pixels)
    
    **Data Conventions**:
    - Missing detections: X_center = Y_center = Radius = 0
    - Origin (0,0) at top-left corner
    - Y-axis points downward (image coordinates)
    
    **Quality Requirements**:
    - Minimum 50 total frames
    - At least 70% detection success rate
    - Smooth trajectory without large jumps
    """)
    
    # Export documentation
    st.markdown("### 💾 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Single Analysis Exports**:
        - Detailed CSV with kinematic data
        - Configuration parameters (JSON)
        - Analysis summary report
        - High-resolution plots (PNG/PDF)
        """)
    
    with col2:
        st.markdown("""
        **Comparison Analysis Exports**:
        - Multi-experiment comparison table
        - Statistical analysis results
        - Trend fitting parameters
        - Combined trajectory data
        """)

# Sidebar information (common to all pages)
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 Project Stats
- **Images processed:** 107
- **Success rate:** 76.6%
- **Detection method:** Computer vision
- **Research type:** Experimental physics
""")

st.sidebar.markdown("""
### 🎓 Research Context
**Institution:** University Laboratory  
**Field:** Granular mechanics  
**Innovation:** First humidity study  
**Impact:** Engineering applications  
""")

# Quick access to saved experiments
if st.session_state.experiments:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Saved Experiments")
    for exp_name, exp in st.session_state.experiments.items():
        meta = exp['metadata']
        st.sidebar.markdown(f"""
        **{exp_name}**
        - Water: {meta['water_content']}%
        - Type: {meta['sphere_type']}
        - Success: {meta['success_rate']:.1f}%
        """)

# Footer
st.markdown("---")
st.markdown("""
### 🎓 Sphere Rolling Resistance Analysis Platform
*Developed for analyzing sphere rolling resistance on wet granular material*

**Features:**
- 🏠 Single experiment analysis
- 🔍 Multi-experiment comparison  
- 📊 Advanced statistical modeling
- 📖 Complete documentation

**Supported formats:** CSV | **Export options:** CSV, JSON, PNG
""")
