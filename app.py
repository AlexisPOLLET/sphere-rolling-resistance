import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sphere Rolling Resistance Analysis",
    page_icon="⚪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for multi-experiment comparison
if 'experiments' not in st.session_state:
    st.session_state.experiments = {}

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
</style>
""", unsafe_allow_html=True)

# Page navigation
page = st.sidebar.selectbox("📋 Select Page", [
    "🏠 Single Analysis",
    "🔍 Multi-Experiment Comparison"
])

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

# Function to load data with metadata
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

# Function to load data (original)
@st.cache_data
def load_uploaded_data(uploaded_file):
    """Loads data from uploaded file"""
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
        
        return df, df_valid
    return None, None

# Function to create sample data (original)
def create_sample_data():
    """Creates sample data for demonstration"""
    frames = list(range(1, 108))
    data = []
    
    for frame in frames:
        if frame < 9:
            data.append([frame, 0, 0, 0])
        elif frame in [30, 31]:
            data.append([frame, 0, 0, 0])
        else:
            x = 1240 - (frame - 9) * 12 + np.random.normal(0, 2)
            y = 680 + (frame - 9) * 0.5 + np.random.normal(0, 3)
            radius = 20 + np.random.normal(5, 3)
            radius = max(18, min(35, radius))
            data.append([frame, max(0, x), max(0, y), max(0, radius)])
    
    return pd.DataFrame(data, columns=['Frame', 'X_center', 'Y_center', 'Radius'])

# ==================== SINGLE ANALYSIS PAGE (ORIGINAL) ====================
if page == "🏠 Single Analysis":

    # Main title
    st.markdown("""
    # ⚪ Sphere Rolling Resistance Analysis Platform
    ## 🔬 Complete Analysis Suite for Granular Mechanics Research
    *Upload your data and access our 3 specialized analysis tools*
    """)

    # File upload section
    st.markdown("""
    <div class="upload-section">
        <h2>📂 Upload Your Experimental Data</h2>
        <p>Start by uploading your CSV file with detection results to get a personalized analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Experiment metadata input for saving
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

    # Global variables for data
    df = None
    df_valid = None

    # Load data
    if uploaded_file is not None:
        df, df_valid = load_uploaded_data(uploaded_file)
        if df is not None:
            st.success(f"✅ File loaded successfully! {len(df)} frames detected")
            
            # Add option to save experiment for comparison
            if st.button("💾 Save experiment for comparison"):
                metadata = {
                    'experiment_name': experiment_name,
                    'water_content': water_content,
                    'sphere_type': sphere_type,
                    'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_frames': len(df),
                    'valid_detections': len(df_valid),
                    'success_rate': len(df_valid) / len(df) * 100 if len(df) > 0 else 0
                }
                st.session_state.experiments[experiment_name] = {
                    'data': df,
                    'metadata': metadata
                }
                st.success(f"Experiment '{experiment_name}' saved for comparison!")
    else:
        # Option to use sample data
        if st.button("🔬 Use sample data for demonstration"):
            df = create_sample_data()
            df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
            st.info("📊 Sample data loaded - you can now explore the features")

    # Navigation only if data is loaded
    if df is not None:
        
        # Quick data overview
        st.markdown("### 📊 Overview of Your Data")
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
        
        # Navigation between the 3 codes
        st.markdown("---")
        st.markdown("## 🔧 Choose Your Analysis")
        
        # Sidebar for navigation
        st.sidebar.title("🧭 Navigation")
        analysis_type = st.sidebar.selectbox("Select analysis type:", [
            "📈 Code 1: Trajectory Visualization",
            "📊 Code 2: Krr Analysis",
            "🔬 Code 3: Complete Analysis",
            "📋 Data Overview"
        ])
        
        # === CODE 1: DETECTION AND TRAJECTORY VISUALIZATION ===
        if analysis_type == "📈 Code 1: Trajectory Visualization":
            st.markdown("""
            <div class="section-header">
                <h2>📈 Code 1: Detection and Trajectory Visualization</h2>
                <p>Complete sphere detection system with trajectory analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detection configuration
            st.markdown("### ⚙️ Detection Configuration")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Size Parameters**")
                minR = st.slider("Minimum radius", 10, 30, 18)
                maxR = st.slider("Maximum radius", 25, 50, 35)
                
            with col2:
                st.markdown("**Detection Parameters**")
                bw_threshold = st.slider("Detection threshold", 1, 20, 8)
                min_score = st.slider("Minimum score", 20, 60, 40)
                
            with col3:
                st.markdown("**Shape Parameters**")
                circularity_min = st.slider("Minimum circularity", 0.1, 1.0, 0.5)
                max_movement = st.slider("Max movement", 50, 200, 120)
            
            # Visualization of loaded data
            if len(df_valid) > 0:
                st.markdown("### 🎯 Detected Sphere Trajectory")
                
                # Main trajectory plot
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('🛤️ Complete Trajectory', '📍 X Position vs Time', 
                                   '📍 Y Position vs Time', '⚪ Radius Evolution'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Trajectory with color gradient based on time
                fig.add_trace(
                    go.Scatter(x=df_valid['X_center'], y=df_valid['Y_center'],
                              mode='markers+lines', 
                              marker=dict(color=df_valid['Frame'], 
                                        colorscale='viridis', 
                                        size=8,
                                        colorbar=dict(title="Frame")),
                              line=dict(width=2),
                              name='Trajectory'),
                    row=1, col=1
                )
                
                # X Position
                fig.add_trace(
                    go.Scatter(x=df_valid['Frame'], y=df_valid['X_center'],
                              mode='lines+markers', 
                              line=dict(color='#3498db', width=3),
                              name='X Position'),
                    row=1, col=2
                )
                
                # Y Position
                fig.add_trace(
                    go.Scatter(x=df_valid['Frame'], y=df_valid['Y_center'],
                              mode='lines+markers',
                              line=dict(color='#e74c3c', width=3),
                              name='Y Position'),
                    row=2, col=1
                )
                
                # Detected radius
                fig.add_trace(
                    go.Scatter(x=df_valid['Frame'], y=df_valid['Radius'],
                              mode='lines+markers',
                              line=dict(color='#2ecc71', width=3),
                              name='Radius'),
                    row=2, col=2
                )
                
                fig.update_layout(height=800, showlegend=False,
                                 title_text="Complete Detection Analysis")
                
                # Reverse Y axis for trajectory (image coordinates)
                fig.update_yaxes(autorange="reversed", row=1, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detection statistics
                st.markdown("### 📊 Detection Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_distance = np.sqrt(
                        (df_valid['X_center'].iloc[-1] - df_valid['X_center'].iloc[0])**2 + 
                        (df_valid['Y_center'].iloc[-1] - df_valid['Y_center'].iloc[0])**2
                    )
                    st.metric("Total Distance", f"{total_distance:.1f} px")
                    
                with col2:
                    if len(df_valid) > 1:
                        dx = df_valid['X_center'].diff()
                        dy = df_valid['Y_center'].diff()
                        speed = np.sqrt(dx**2 + dy**2)
                        avg_speed = speed.mean()
                        st.metric("Average Speed", f"{avg_speed:.2f} px/frame")
                        
                with col3:
                    vertical_displacement = abs(df_valid['Y_center'].iloc[-1] - df_valid['Y_center'].iloc[0])
                    st.metric("Vertical Displacement", f"{vertical_displacement:.1f} px")
                    
                with col4:
                    avg_radius = df_valid['Radius'].mean()
                    radius_std = df_valid['Radius'].std()
                    st.metric("Average Radius", f"{avg_radius:.1f} ± {radius_std:.1f} px")
                
                # Detection quality analysis
                st.markdown("### 🔍 Detection Quality")
                
                # Radius distribution plot
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_radius = px.histogram(df_valid, x='Radius', nbins=15,
                                             title="Distribution of Detected Radii",
                                             labels={'Radius': 'Radius (pixels)', 'count': 'Frequency'})
                    fig_radius.add_vline(x=minR, line_dash="dash", line_color="red", 
                                        annotation_text=f"Min: {minR}")
                    fig_radius.add_vline(x=maxR, line_dash="dash", line_color="red", 
                                        annotation_text=f"Max: {maxR}")
                    st.plotly_chart(fig_radius, use_container_width=True)
                    
                with col2:
                    # Movement continuity analysis
                    if len(df_valid) > 1:
                        dx = df_valid['X_center'].diff()
                        dy = df_valid['Y_center'].diff()
                        movement = np.sqrt(dx**2 + dy**2)
                        # Remove NaN values and create correct DataFrame
                        movement_clean = movement.dropna()
                        frames_clean = df_valid['Frame'][1:len(movement_clean)+1]
                        
                        # Create plot with go.Scatter instead of px.line
                        fig_movement = go.Figure()
                        fig_movement.add_trace(go.Scatter(
                            x=frames_clean, 
                            y=movement_clean,
                            mode='lines+markers',
                            name='Movement',
                            line=dict(color='blue', width=2)
                        ))
                        fig_movement.add_hline(y=max_movement, line_dash="dash", line_color="red",
                                              annotation_text=f"Max allowed: {max_movement}")
                        fig_movement.update_layout(
                            title="Inter-Frame Movement",
                            xaxis_title="Frame",
                            yaxis_title="Displacement (pixels)",
                            height=400
                        )
                        st.plotly_chart(fig_movement, use_container_width=True)
            
            # Information about detection algorithm
            st.markdown("### 🧠 Detection Algorithm")
            st.markdown(f"""
            **Method used:** Circle detection by background subtraction
            
            **Main steps:**
            1. **Background creation**: Average of 150 reference images
            2. **Subtraction**: Elimination of static background
            3. **Thresholding**: Binarization with adaptive threshold
            4. **Morphology**: Contour cleaning
            5. **Detection**: Search for circular contours
            6. **Validation**: Filtering by size, shape and continuity
            
            **Quality criteria:**
            - Size: {minR} ≤ radius ≤ {maxR} pixels
            - Shape: Circularity ≥ {circularity_min}
            - Continuity: Movement ≤ {max_movement} pixels/frame
            - Score: Overall quality ≥ {min_score}
            """)
        
        # === CODE 2: KRR ANALYSIS ===
        elif analysis_type == "📊 Code 2: Krr Analysis":
            st.markdown("""
            <div class="section-header">
                <h2>📊 Code 2: Rolling Resistance Coefficient (Krr) Analysis</h2>
                <p>Complete physical calculations to determine the Krr coefficient</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sphere parameters
            st.markdown("### 🔵 Sphere Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sphere_radius_mm = st.number_input("Sphere radius (mm)", value=15.0, min_value=1.0, max_value=50.0)
                sphere_mass_g = st.number_input("Sphere mass (g)", value=10.0, min_value=0.1, max_value=1000.0)
                
            with col2:
                sphere_type = st.selectbox("Sphere type", ["Solid (j=2/5)", "Hollow (j=2/3)"])
                j_value = 2/5 if "Solid" in sphere_type else 2/3
                
                # Density calculation
                volume_mm3 = (4/3) * np.pi * sphere_radius_mm**3
                volume_m3 = volume_mm3 * 1e-9
                mass_kg = sphere_mass_g * 1e-3
                density_kg_m3 = mass_kg / volume_m3
                st.metric("Density", f"{density_kg_m3:.0f} kg/m³")
                
            with col3:
                st.metric("Inertia factor j", f"{j_value:.3f}")
                st.metric("Factor (1+j)⁻¹", f"{1/(1+j_value):.4f}")
            
            # Experimental parameters
            st.markdown("### 📐 Experimental Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fps = st.number_input("Camera FPS", value=250.0, min_value=1.0, max_value=1000.0)
                angle_deg = st.number_input("Inclination angle (°)", value=15.0, min_value=0.1, max_value=45.0)
                
            with col2:
                # Automatic calibration based on detected radius
                if len(df_valid) > 0:
                    avg_radius_pixels = df_valid['Radius'].mean()
                    auto_calibration = avg_radius_pixels / sphere_radius_mm
                    st.metric("Auto calibration", f"{auto_calibration:.2f} px/mm")
                    
                    use_auto_cal = st.checkbox("Use automatic calibration", value=True)
                    if use_auto_cal:
                        pixels_per_mm = auto_calibration
                    else:
                        pixels_per_mm = st.number_input("Calibration (px/mm)", value=auto_calibration, min_value=0.1)
                else:
                    pixels_per_mm = st.number_input("Calibration (px/mm)", value=5.0, min_value=0.1)
                    
            with col3:
                water_content_analysis = st.number_input("Water content (%)", value=0.0, min_value=0.0, max_value=100.0, key="krr_water")
                
            # Kinematic calculations and Krr
            if len(df_valid) > 10:
                st.markdown("### 🧮 Kinematic Calculations")
                
                # Unit conversion
                dt = 1 / fps  # s
                
                # Positions in meters
                x_mm = df_valid['X_center'].values / pixels_per_mm
                y_mm = df_valid['Y_center'].values / pixels_per_mm
                x_m = x_mm / 1000
                y_m = y_mm / 1000
                
                # Time
                t = np.arange(len(df_valid)) * dt
                
                # Velocities
                vx = np.gradient(x_m, dt)
                vy = np.gradient(y_m, dt)
                v_magnitude = np.sqrt(vx**2 + vy**2)
                
                # Initial and final velocities (average over a few points)
                n_avg = min(3, len(v_magnitude)//4)
                v0 = np.mean(v_magnitude[:n_avg])
                vf = np.mean(v_magnitude[-n_avg:])
                
                # Total distance
                distances = np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2)
                total_distance = np.sum(distances)
                
                # Calculate Krr coefficient
                g = 9.81  # m/s²
                if total_distance > 0:
                    krr = (v0**2 - vf**2) / (2 * g * total_distance)
                    
                    # Effective friction coefficient
                    angle_rad = np.radians(angle_deg)
                    mu_eff = krr + np.tan(angle_rad)
                    
                    # Display results
                    st.markdown("### 📈 Krr Results")
                    
                    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                    
                    with result_col1:
                        st.metric("V₀ (initial velocity)", f"{v0*1000:.1f} mm/s")
                        st.caption(f"{v0:.4f} m/s")
                        
                    with result_col2:
                        st.metric("Vf (final velocity)", f"{vf*1000:.1f} mm/s") 
                        st.caption(f"{vf:.4f} m/s")
                        
                    with result_col3:
                        st.metric("Total distance", f"{total_distance*1000:.1f} mm")
                        st.caption(f"{total_distance:.4f} m")
                        
                    with result_col4:
                        st.metric("**Krr Coefficient**", f"{krr:.6f}")
                        if 0.03 <= krr <= 0.10:
                            st.success("✅ Consistent with Van Wal (2017)")
                        elif krr < 0:
                            st.error("⚠️ Negative Krr - sphere accelerating")
                        else:
                            st.warning("⚠️ Different from literature")
                    
                    # Quick trajectory visualization
                    st.markdown("### 🎯 Trajectory and Velocity Profile")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_traj = px.scatter(df_valid, x='X_center', y='Y_center', 
                                           color='Frame', size='Radius',
                                           title="Sphere Trajectory")
                        fig_traj.update_yaxes(autorange="reversed")
                        st.plotly_chart(fig_traj, use_container_width=True)
                    
                    with col2:
                        fig_vel = go.Figure()
                        fig_vel.add_trace(go.Scatter(x=t, y=v_magnitude*1000, 
                                                   mode='lines+markers',
                                                   name='Velocity',
                                                   line=dict(color='blue', width=2)))
                        fig_vel.update_layout(
                            title="Velocity vs Time",
                            xaxis_title="Time (s)",
                            yaxis_title="Velocity (mm/s)"
                        )
                        st.plotly_chart(fig_vel, use_container_width=True)
                
                else:
                    st.error("❌ Zero distance traveled - impossible to calculate Krr")
            else:
                st.warning("⚠️ Not enough valid data for Krr analysis")
        
        # === CODE 3: ADVANCED AND COMPLETE ANALYSIS ===
        elif analysis_type == "🔬 Code 3: Complete Analysis":
            st.markdown("""
            <div class="section-header">
                <h2>🔬 Code 3: Advanced and Complete Kinematic Analysis</h2>
                <p>In-depth analysis with debug and advanced metrics</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Data verification
            st.markdown("### 🔍 Data Verification")
            if len(df_valid) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Valid data", f"{len(df_valid)} frames")
                    st.metric("Success rate", f"{len(df_valid)/len(df)*100:.1f}%")
                    
                with col2:
                    radius_range = df_valid['Radius'].max() - df_valid['Radius'].min()
                    st.metric("Radius variation", f"{radius_range:.1f} px")
                    st.metric("First detection", f"Frame {df_valid['Frame'].min()}")
                    
                with col3:
                    st.metric("Last detection", f"Frame {df_valid['Frame'].max()}")
                    duration_frames = df_valid['Frame'].max() - df_valid['Frame'].min()
                    st.metric("Tracking duration", f"{duration_frames} frames")
                
                # Parameters for advanced analysis
                st.markdown("### ⚙️ Advanced Analysis Parameters")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Sphere Parameters**")
                    mass_g = st.number_input("Mass (g)", value=10.0, min_value=0.1, key="adv_mass")
                    radius_mm = st.number_input("Radius (mm)", value=15.0, min_value=1.0, key="adv_radius")
                    sphere_type = st.selectbox("Type", ["Solid", "Hollow"], key="adv_type")
                    j_factor = 2/5 if sphere_type == "Solid" else 2/3
                    
                with col2:
                    st.markdown("**Experimental Parameters**")
                    fps = st.number_input("FPS", value=250.0, min_value=1.0, key="adv_fps")
                    angle_deg = st.number_input("Angle (°)", value=15.0, min_value=0.1, key="adv_angle")
                    
                    # Automatic calibration
                    if len(df_valid) > 0:
                        avg_radius_px = df_valid['Radius'].mean()
                        auto_cal = avg_radius_px / radius_mm
                        st.metric("Auto calibration", f"{auto_cal:.2f} px/mm")
                        pixels_per_mm = auto_cal
                    
                with col3:
                    st.markdown("**Data Filtering**")
                    use_smoothing = st.checkbox("Data smoothing", value=True)
                    smooth_window = st.slider("Smoothing window", 3, 11, 5, step=2)
                    remove_outliers = st.checkbox("Remove outliers", value=True)
                    
                # Advanced kinematic calculations
                if st.button("🚀 Launch Complete Analysis"):
                    
                    st.markdown("### 🧮 Advanced Kinematic Calculations")
                    
                    # Data extraction and preparation
                    t = np.arange(len(df_valid)) / fps
                    x_mm = df_valid['X_center'].values / pixels_per_mm
                    y_mm = df_valid['Y_center'].values / pixels_per_mm
                    x_m = x_mm / 1000
                    y_m = y_mm / 1000
                    
                    # Remove outliers if requested
                    if remove_outliers:
                        # Simple outlier detection by standard deviation
                        def remove_outliers_1d(data, threshold=2):
                            mean_val = np.mean(data)
                            std_val = np.std(data)
                            mask = np.abs(data - mean_val) < threshold * std_val
                            return mask
                        
                        mask_x = remove_outliers_1d(x_m)
                        mask_y = remove_outliers_1d(y_m)
                        mask = mask_x & mask_y
                        
                        t = t[mask]
                        x_m = x_m[mask]
                        y_m = y_m[mask]
                        x_mm = x_mm[mask]
                        y_mm = y_mm[mask]
                        
                        st.info(f"🔧 Outliers removed: {np.sum(~mask)} points")
                    
                    # Calculate velocities with optional smoothing
                    dt = np.mean(np.diff(t)) if len(t) > 1 else 1/fps
                    
                    if use_smoothing and len(x_m) >= smooth_window:
                        try:
                            from scipy.signal import savgol_filter
                            x_smooth = savgol_filter(x_m, smooth_window, 2)
                            y_smooth = savgol_filter(y_m, smooth_window, 2)
                            vx = np.gradient(x_smooth, dt)
                            vy = np.gradient(y_smooth, dt)
                            st.success(f"✅ Data smoothed with window {smooth_window}")
                        except:
                            vx = np.gradient(x_m, dt)
                            vy = np.gradient(y_m, dt)
                            st.warning("⚠️ Smoothing failed, using raw data")
                    else:
                        vx = np.gradient(x_m, dt)
                        vy = np.gradient(y_m, dt)
                    
                    v_magnitude = np.sqrt(vx**2 + vy**2)
                    
                    # Acceleration
                    acceleration = np.gradient(v_magnitude, dt)
                    
                    # Physical parameters
                    mass_kg = mass_g / 1000
                    radius_m = radius_mm / 1000
                    angle_rad = np.radians(angle_deg)
                    g = 9.81
                    
                    # Forces and energies
                    F_resistance = mass_kg * acceleration
                    F_gravity = mass_kg * g * np.sin(angle_rad)
                    
                    # Kinetic energies
                    E_trans = 0.5 * mass_kg * v_magnitude**2
                    I = j_factor * mass_kg * radius_m**2
                    omega = v_magnitude / radius_m
                    E_rot = 0.5 * I * omega**2
                    E_total = E_trans + E_rot
                    
                    # Power and instantaneous Krr
                    P_resistance = np.abs(F_resistance * v_magnitude)
                    Krr_inst = np.abs(F_resistance) / (mass_kg * g)
                    
                    # Global metrics
                    avg_krr = np.mean(Krr_inst)
                    energy_dissipated = (E_total[0] - E_total[-1]) * 1000  # mJ
                    avg_power = np.mean(P_resistance) * 1000  # mW
                    
                    # Display results
                    st.markdown("### 📊 Advanced Analysis Results")
                    
                    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                    
                    with result_col1:
                        st.metric("Average Krr", f"{avg_krr:.6f}")
                        st.metric("Median Krr", f"{np.median(Krr_inst):.6f}")
                        
                    with result_col2:
                        st.metric("Dissipated Energy", f"{energy_dissipated:.2f} mJ")
                        st.metric("Average Power", f"{avg_power:.2f} mW")
                        
                    with result_col3:
                        st.metric("Max Velocity", f"{np.max(v_magnitude)*1000:.1f} mm/s")
                        st.metric("Min Velocity", f"{np.min(v_magnitude)*1000:.1f} mm/s")
                        
                    with result_col4:
                        st.metric("Max Accel.", f"{np.max(np.abs(acceleration))*1000:.1f} mm/s²")
                        total_distance = np.sum(np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2))
                        st.metric("Total Distance", f"{total_distance*1000:.1f} mm")
                    
                    # Advanced plots
                    st.markdown("### 📈 Advanced Visualizations")
                    
                    # Main figure with 6 subplots
                    fig_advanced = make_subplots(
                        rows=3, cols=2,
                        subplot_titles=('Smoothed Velocity vs Time', 'Acceleration vs Time',
                                       'Kinetic Energies', 'Instantaneous Krr',
                                       'Resistance Power', 'Forces'),
                        vertical_spacing=0.08
                    )
                    
                    # 1. Velocity
                    fig_advanced.add_trace(
                        go.Scatter(x=t, y=v_magnitude*1000, mode='lines', name='Velocity',
                                  line=dict(color='blue', width=2)),
                        row=1, col=1
                    )
                    
                    # 2. Acceleration
                    fig_advanced.add_trace(
                        go.Scatter(x=t, y=acceleration*1000, mode='lines', name='Acceleration',
                                  line=dict(color='red', width=2)),
                        row=1, col=2
                    )
                    
                    # 3. Energies
                    fig_advanced.add_trace(
                        go.Scatter(x=t, y=E_trans*1000, mode='lines', name='Translation',
                                  line=dict(color='blue', width=2)),
                        row=2, col=1
                    )
                    fig_advanced.add_trace(
                        go.Scatter(x=t, y=E_rot*1000, mode='lines', name='Rotation',
                                  line=dict(color='red', width=2)),
                        row=2, col=1
                    )
                    fig_advanced.add_trace(
                        go.Scatter(x=t, y=E_total*1000, mode='lines', name='Total',
                                  line=dict(color='black', width=3)),
                        row=2, col=1
                    )
                    
                    # 4. Instantaneous Krr
                    fig_advanced.add_trace(
                        go.Scatter(x=t, y=Krr_inst, mode='lines', name='Krr',
                                  line=dict(color='purple', width=2)),
                        row=2, col=2
                    )
                    fig_advanced.add_hline(y=avg_krr, line_dash="dash", line_color="orange", row=2, col=2)
                    
                    # 5. Power
                    fig_advanced.add_trace(
                        go.Scatter(x=t, y=P_resistance*1000, mode='lines', name='Power',
                                  line=dict(color='green', width=2)),
                        row=3, col=1
                    )
                    
                    # 6. Forces
                    fig_advanced.add_trace(
                        go.Scatter(x=t, y=F_resistance*1000, mode='lines', name='F_resistance',
                                  line=dict(color='red', width=2)),
                        row=3, col=2
                    )
                    fig_advanced.add_hline(y=F_gravity*1000, line_dash="dash", line_color="blue", row=3, col=2)
                    
                    # Update axes
                    fig_advanced.update_xaxes(title_text="Time (s)")
                    fig_advanced.update_yaxes(title_text="Velocity (mm/s)", row=1, col=1)
                    fig_advanced.update_yaxes(title_text="Acceleration (mm/s²)", row=1, col=2)
                    fig_advanced.update_yaxes(title_text="Energy (mJ)", row=2, col=1)
                    fig_advanced.update_yaxes(title_text="Krr Coefficient", row=2, col=2)
                    fig_advanced.update_yaxes(title_text="Power (mW)", row=3, col=1)
                    fig_advanced.update_yaxes(title_text="Force (mN)", row=3, col=2)
                    
                    fig_advanced.update_layout(height=900, showlegend=False)
                    st.plotly_chart(fig_advanced, use_container_width=True)
                    
                    # Download button for detailed data
                    detailed_data = pd.DataFrame({
                        'time_s': t,
                        'x_mm': x_mm,
                        'y_mm': y_mm,
                        'vx_ms': vx,
                        'vy_ms': vy,
                        'v_magnitude_ms': v_magnitude,
                        'acceleration_ms2': acceleration,
                        'F_resistance_N': F_resistance,
                        'E_total_J': E_total,
                        'P_resistance_W': P_resistance,
                        'Krr_instantaneous': Krr_inst
                    })
                    
                    csv_data = detailed_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download detailed data (CSV)",
                        data=csv_data,
                        file_name="advanced_kinetic_analysis.csv",
                        mime="text/csv"
                    )
                    
            else:
                st.error("❌ No valid data for advanced analysis")
        
        # === DATA OVERVIEW ===
        else:  # Data overview
            st.markdown("""
            <div class="section-header">
                <h2>📋 Overview of Your Data</h2>
                <p>Exploration and validation of data quality</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display first rows
            st.markdown("### 📊 Data Preview")
            st.dataframe(df.head(10))
            
            # Descriptive statistics
            st.markdown("### 📈 Descriptive Statistics")
            st.dataframe(df_valid.describe())
            
            # Distribution plots
            if len(df_valid) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = px.histogram(df_valid, x='Radius', title="Radius Distribution")
                    st.plotly_chart(fig1, use_container_width=True)
                    
                with col2:
                    detection_status = df['X_center'] != 0
                    fig2 = px.pie(values=[detection_status.sum(), (~detection_status).sum()],
                                 names=['Detected', 'Not detected'],
                                 title="Detection Distribution")
                    st.plotly_chart(fig2, use_container_width=True)

    else:
        # Message if no data is loaded
        st.markdown("""
        ## 🚀 To get started:
        
        1. **📂 Upload your CSV file** with your experimental data
        2. **Or click "Use sample data for demonstration"** to explore the features
        3. **🔧 Choose the analysis** that interests you from the menu
        
        ### 📋 Expected file format:
        Your CSV must contain the following columns:
        - `Frame`: Image number
        - `X_center`: X position of sphere center
        - `Y_center`: Y position of sphere center  
        - `Radius`: Detected sphere radius
        
        ### 🔧 The 3 Integrated Codes:
        - **Code 1**: Trajectory visualization
        - **Code 2**: Krr analysis (resistance coefficient)
        - **Code 3**: Complete and advanced analysis
        """)

# ==================== MULTI-EXPERIMENT COMPARISON PAGE (NEW) ====================
elif page == "🔍 Multi-Experiment Comparison":
    
    st.markdown("""
    # 🔍 Multi-Experiment Comparison
    ## Compare multiple experiments to analyze the effect of different parameters
    """)
    
    # Check if experiments are available
    if not st.session_state.experiments:
        st.warning("⚠️ No experiments available for comparison. Please load some experiments from the Single Analysis page first.")
        
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
                    
                    # Add trend line
                    if len(comp_df) >= 2:
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
                    # Create velocity comparison chart
                    fig_vel = go.Figure()
                    
                    fig_vel.add_trace(go.Scatter(
                        x=comp_df['Water_Content'],
                        y=comp_df['Initial_Velocity'] * 1000,  # Convert to mm/s
                        mode='markers+lines',
                        name='Initial Velocity',
                        marker=dict(color='blue', size=10)
                    ))
                    
                    fig_vel.add_trace(go.Scatter(
                        x=comp_df['Water_Content'],
                        y=comp_df['Final_Velocity'] * 1000,  # Convert to mm/s
                        mode='markers+lines',
                        name='Final Velocity',
                        marker=dict(color='red', size=10)
                    ))
                    
                    fig_vel.update_layout(
                        title="🏃 Velocity Comparison",
                        xaxis_title="Water Content (%)",
                        yaxis_title="Velocity (mm/s)"
                    )
                    st.plotly_chart(fig_vel, use_container_width=True)
            
            with col2:
                if comp_df['Distance'].notna().any():
                    fig_dist = px.bar(comp_df, x='Experiment', y='Distance',
                                    color='Water_Content',
                                    title="📏 Distance Traveled Comparison")
                    fig_dist.update_xaxes(tickangle=45)
                    fig_dist.update_layout(yaxis_title="Distance (m)")
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            # Trajectory overlay
            if trajectory_data:
                st.markdown("### 🛤️ Trajectory Overlay Comparison")
                
                combined_traj = pd.concat(trajectory_data, ignore_index=True)
                fig_overlay = px.scatter(combined_traj, x='X_center', y='Y_center',
                                       color='Experiment', 
                                       title="Trajectory Comparison - All Experiments",
                                       opacity=0.7)
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

# Footer
st.markdown("---")
st.markdown("""
### 🎓 Sphere Rolling Resistance Analysis Platform
*Developed for analyzing sphere rolling resistance on wet granular material*
""")

# Sidebar with information
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
**Institution:** Department of Cosmic Earth Science, Graduate School of Science, Osaka University  
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
        """)
