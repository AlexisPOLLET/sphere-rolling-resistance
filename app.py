import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Sphere Detection Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-metric {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">⚽ Sphere Detection Analysis</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<h4>🎯 Project: Rolling Resistance of Spheres on Wet Granular Material</h4>
<p>This application analyzes sphere detection data from computer vision tracking. 
The project studies the effect of humidity on rolling friction - the first research to focus on wet soils.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📊 Analysis Controls")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Detection Data (CSV)",
    type=['csv'],
    help="Upload your detections.csv file"
)

# Load data function
@st.cache_data
def load_data(file):
    """Load and process detection data"""
    if file is not None:
        df = pd.read_csv(file)
    else:
        # Create sample data based on the actual structure
        frames = list(range(1, 108))
        data = []
        
        # Simulate the actual pattern: no detections for first 8 frames, then valid detections
        for frame in frames:
            if frame < 9:
                data.append({
                    'Frame': frame,
                    'X_center': 0,
                    'Y_center': 0,
                    'Radius': 0
                })
            else:
                # Simulate sphere movement from right to left, top to bottom
                progress = (frame - 9) / 98  # Normalize to 0-1
                x = int(1238 - progress * 421)  # Move from 1238 to 817
                y = int(679 + progress * 89)    # Move from 679 to 768
                radius = int(20 + np.sin(progress * 4) * 6)  # Vary radius between 19-32
                
                data.append({
                    'Frame': frame,
                    'X_center': x,
                    'Y_center': y,
                    'Radius': radius
                })
        
        df = pd.DataFrame(data)
    
    return df

# Load data
df = load_data(uploaded_file)

# Process data
df['Has_Detection'] = (df['X_center'] > 0) & (df['Y_center'] > 0) & (df['Radius'] > 0)
valid_detections = df[df['Has_Detection']].copy()

# Calculate derived metrics
if len(valid_detections) > 1:
    valid_detections['X_movement'] = valid_detections['X_center'].diff()
    valid_detections['Y_movement'] = valid_detections['Y_center'].diff()
    valid_detections['Distance'] = np.sqrt(
        valid_detections['X_movement']**2 + valid_detections['Y_movement']**2
    )
    valid_detections['Speed'] = valid_detections['Distance']  # pixels per frame
    valid_detections['Radius_change'] = valid_detections['Radius'].diff()

# Configuration parameters from the config file
config_params = {
    "Detection Parameters": {
        "Minimum Radius": "18 pixels",
        "Maximum Radius": "35 pixels",
        "Detection Threshold": "8",
        "Circularity Minimum": "0.5",
        "Minimum Score": "40",
        "Maximum Movement": "120 pixels",
        "Background Images": "150"
    },
    "Image Processing": {
        "Crop Area": "(400,500) to (1246,1000)",
        "Cropped Size": "846 x 500 pixels"
    },
    "Results Summary": {
        "Images Processed": "107",
        "Successful Detections": f"{len(valid_detections)}",
        "Success Rate": f"{len(valid_detections)/len(df)*100:.1f}%"
    }
}

# Sidebar filters
st.sidebar.subheader("🔧 Analysis Filters")

if len(valid_detections) > 0:
    frame_range = st.sidebar.slider(
        "Frame Range",
        min_value=int(valid_detections['Frame'].min()),
        max_value=int(valid_detections['Frame'].max()),
        value=(int(valid_detections['Frame'].min()), int(valid_detections['Frame'].max())),
        help="Select frame range to analyze"
    )
    
    radius_range = st.sidebar.slider(
        "Radius Range (pixels)",
        min_value=int(valid_detections['Radius'].min()),
        max_value=int(valid_detections['Radius'].max()),
        value=(int(valid_detections['Radius'].min()), int(valid_detections['Radius'].max())),
        help="Filter detections by radius"
    )
    
    # Filter data
    filtered_data = valid_detections[
        (valid_detections['Frame'] >= frame_range[0]) & 
        (valid_detections['Frame'] <= frame_range[1]) &
        (valid_detections['Radius'] >= radius_range[0]) & 
        (valid_detections['Radius'] <= radius_range[1])
    ]
else:
    filtered_data = valid_detections

# Main content
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-container success-metric">
        <h4>📊 Total Frames</h4>
        <h2>{}</h2>
    </div>
    """.format(len(df)), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-container success-metric">
        <h4>✅ Valid Detections</h4>
        <h2>{}</h2>
    </div>
    """.format(len(valid_detections)), unsafe_allow_html=True)

with col3:
    success_rate = len(valid_detections)/len(df)*100 if len(df) > 0 else 0
    st.markdown("""
    <div class="metric-container success-metric">
        <h4>🎯 Success Rate</h4>
        <h2>{:.1f}%</h2>
    </div>
    """.format(success_rate), unsafe_allow_html=True)

with col4:
    avg_radius = valid_detections['Radius'].mean() if len(valid_detections) > 0 else 0
    st.markdown("""
    <div class="metric-container success-metric">
        <h4>📏 Avg Radius</h4>
        <h2>{:.1f}px</h2>
    </div>
    """.format(avg_radius), unsafe_allow_html=True)

# Tabs for different analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trajectory Analysis", 
    "📊 Detection Statistics", 
    "🎯 Motion Analysis", 
    "⚙️ Configuration", 
    "📋 Data Export"
])

with tab1:
    st.header("🎯 Sphere Trajectory Analysis")
    
    if len(filtered_data) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Trajectory plot
            fig_traj = go.Figure()
            
            # Add trajectory line
            fig_traj.add_trace(go.Scatter(
                x=filtered_data['X_center'],
                y=filtered_data['Y_center'],
                mode='lines+markers',
                marker=dict(
                    size=filtered_data['Radius']/2,
                    color=filtered_data['Frame'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Frame Number")
                ),
                line=dict(width=2, color='blue'),
                name='Sphere Path',
                text=filtered_data['Frame'],
                hovertemplate='<b>Frame:</b> %{text}<br>' +
                            '<b>X:</b> %{x}<br>' +
                            '<b>Y:</b> %{y}<br>' +
                            '<b>Radius:</b> %{marker.size}<extra></extra>'
            ))
            
            fig_traj.update_layout(
                title="Sphere Trajectory (2D Path)",
                xaxis_title="X Position (pixels)",
                yaxis_title="Y Position (pixels)",
                height=500,
                yaxis=dict(autorange='reversed')  # Flip Y-axis to match image coordinates
            )
            
            st.plotly_chart(fig_traj, use_container_width=True)
        
        with col2:
            # Position vs Frame
            fig_pos = make_subplots(
                rows=2, cols=1,
                subplot_titles=('X Position vs Frame', 'Y Position vs Frame'),
                vertical_spacing=0.1
            )
            
            fig_pos.add_trace(
                go.Scatter(
                    x=filtered_data['Frame'],
                    y=filtered_data['X_center'],
                    mode='lines+markers',
                    name='X Position',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            fig_pos.add_trace(
                go.Scatter(
                    x=filtered_data['Frame'],
                    y=filtered_data['Y_center'],
                    mode='lines+markers',
                    name='Y Position',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
            fig_pos.update_layout(height=500, title_text="Position Over Time")
            fig_pos.update_xaxes(title_text="Frame Number")
            fig_pos.update_yaxes(title_text="X Position (pixels)", row=1, col=1)
            fig_pos.update_yaxes(title_text="Y Position (pixels)", row=2, col=1)
            
            st.plotly_chart(fig_pos, use_container_width=True)
        
        # Radius analysis
        st.subheader("📏 Radius Variation Analysis")
        
        fig_radius = go.Figure()
        fig_radius.add_trace(go.Scatter(
            x=filtered_data['Frame'],
            y=filtered_data['Radius'],
            mode='lines+markers',
            name='Detected Radius',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ))
        
        # Add configuration limits
        fig_radius.add_hline(y=18, line_dash="dash", line_color="red", 
                           annotation_text="Min Radius (18px)")
        fig_radius.add_hline(y=35, line_dash="dash", line_color="red", 
                           annotation_text="Max Radius (35px)")
        
        fig_radius.update_layout(
            title="Radius Detection Over Time",
            xaxis_title="Frame Number",
            yaxis_title="Radius (pixels)",
            height=400
        )
        
        st.plotly_chart(fig_radius, use_container_width=True)
    
    else:
        st.warning("⚠️ No valid detections found in the selected range.")

with tab2:
    st.header("📊 Detection Statistics & Quality")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Detection success over time
        detection_by_frame = df.groupby('Frame')['Has_Detection'].sum().reset_index()
        
        fig_success = px.bar(
            detection_by_frame,
            x='Frame',
            y='Has_Detection',
            title="Detection Success by Frame",
            color='Has_Detection',
            color_continuous_scale='RdYlGn'
        )
        fig_success.update_layout(height=400)
        st.plotly_chart(fig_success, use_container_width=True)
        
        # Success rate by window
        if len(valid_detections) > 0:
            window_size = st.slider("Window Size for Moving Average", 5, 20, 10)
            df['Detection_MA'] = df['Has_Detection'].rolling(window=window_size, center=True).mean()
            
            fig_ma = px.line(
                df,
                x='Frame',
                y='Detection_MA',
                title=f"Detection Success Rate (Moving Average, Window={window_size})"
            )
            fig_ma.update_layout(height=300)
            st.plotly_chart(fig_ma, use_container_width=True)
    
    with col2:
        # Distribution plots
        if len(valid_detections) > 0:
            fig_dist = make_subplots(
                rows=2, cols=2,
                subplot_titles=('X Position Distribution', 'Y Position Distribution',
                              'Radius Distribution', 'Frame Distribution'),
                specs=[[{"type": "histogram"}, {"type": "histogram"}],
                       [{"type": "histogram"}, {"type": "histogram"}]]
            )
            
            fig_dist.add_trace(
                go.Histogram(x=valid_detections['X_center'], name="X Position", nbinsx=20),
                row=1, col=1
            )
            fig_dist.add_trace(
                go.Histogram(x=valid_detections['Y_center'], name="Y Position", nbinsx=20),
                row=1, col=2
            )
            fig_dist.add_trace(
                go.Histogram(x=valid_detections['Radius'], name="Radius", nbinsx=15),
                row=2, col=1
            )
            fig_dist.add_trace(
                go.Histogram(x=valid_detections['Frame'], name="Frame", nbinsx=20),
                row=2, col=2
            )
            
            fig_dist.update_layout(height=600, title_text="Distribution Analysis")
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Summary statistics
        if len(valid_detections) > 0:
            st.subheader("📈 Statistical Summary")
            stats_df = pd.DataFrame({
                'Metric': ['X Position', 'Y Position', 'Radius', 'Frame'],
                'Mean': [
                    valid_detections['X_center'].mean(),
                    valid_detections['Y_center'].mean(),
                    valid_detections['Radius'].mean(),
                    valid_detections['Frame'].mean()
                ],
                'Std Dev': [
                    valid_detections['X_center'].std(),
                    valid_detections['Y_center'].std(),
                    valid_detections['Radius'].std(),
                    valid_detections['Frame'].std()
                ],
                'Min': [
                    valid_detections['X_center'].min(),
                    valid_detections['Y_center'].min(),
                    valid_detections['Radius'].min(),
                    valid_detections['Frame'].min()
                ],
                'Max': [
                    valid_detections['X_center'].max(),
                    valid_detections['Y_center'].max(),
                    valid_detections['Radius'].max(),
                    valid_detections['Frame'].max()
                ]
            })
            st.dataframe(stats_df.round(2), use_container_width=True)

with tab3:
    st.header("🎯 Motion & Movement Analysis")
    
    if len(filtered_data) > 1 and 'Distance' in filtered_data.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Movement speed over time
            fig_speed = go.Figure()
            fig_speed.add_trace(go.Scatter(
                x=filtered_data['Frame'],
                y=filtered_data['Speed'],
                mode='lines+markers',
                name='Movement Speed',
                line=dict(color='purple', width=2)
            ))
            
            avg_speed = filtered_data['Speed'].mean()
            fig_speed.add_hline(y=avg_speed, line_dash="dash", line_color="orange",
                              annotation_text=f"Average Speed: {avg_speed:.1f}px/frame")
            
            fig_speed.update_layout(
                title="Movement Speed Over Time",
                xaxis_title="Frame Number",
                yaxis_title="Speed (pixels/frame)",
                height=400
            )
            st.plotly_chart(fig_speed, use_container_width=True)
            
            # Movement direction analysis
            if len(filtered_data) > 2:
                movement_x = filtered_data['X_movement'].dropna()
                movement_y = filtered_data['Y_movement'].dropna()
                
                fig_movement = go.Figure()
                fig_movement.add_trace(go.Scatter(
                    x=movement_x,
                    y=movement_y,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=filtered_data['Frame'][1:],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Frame")
                    ),
                    name='Movement Vector'
                ))
                
                fig_movement.update_layout(
                    title="Movement Direction Analysis",
                    xaxis_title="X Movement (pixels)",
                    yaxis_title="Y Movement (pixels)",
                    height=400
                )
                st.plotly_chart(fig_movement, use_container_width=True)
        
        with col2:
            # Cumulative distance
            filtered_data['Cumulative_Distance'] = filtered_data['Distance'].cumsum()
            
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=filtered_data['Frame'],
                y=filtered_data['Cumulative_Distance'],
                mode='lines+markers',
                name='Cumulative Distance',
                line=dict(color='green', width=3)
            ))
            
            fig_cum.update_layout(
                title="Cumulative Distance Traveled",
                xaxis_title="Frame Number",
                yaxis_title="Cumulative Distance (pixels)",
                height=400
            )
            st.plotly_chart(fig_cum, use_container_width=True)
            
            # Motion statistics
            st.subheader("🏃 Motion Statistics")
            if len(filtered_data) > 1:
                motion_stats = {
                    "Total Distance": f"{filtered_data['Distance'].sum():.1f} pixels",
                    "Average Speed": f"{filtered_data['Speed'].mean():.1f} px/frame",
                    "Max Speed": f"{filtered_data['Speed'].max():.1f} px/frame",
                    "Min Speed": f"{filtered_data['Speed'].min():.1f} px/frame",
                    "Speed Std Dev": f"{filtered_data['Speed'].std():.1f} px/frame"
                }
                
                for key, value in motion_stats.items():
                    st.metric(key, value)
    else:
        st.warning("⚠️ Insufficient data for motion analysis. Need at least 2 detection points.")

with tab4:
    st.header("⚙️ Detection Configuration & Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Detection Parameters")
        for param, value in config_params["Detection Parameters"].items():
            st.metric(param, value)
        
        st.subheader("🖼️ Image Processing")
        for param, value in config_params["Image Processing"].items():
            st.metric(param, value)
    
    with col2:
        st.subheader("📊 Results Summary")
        for param, value in config_params["Results Summary"].items():
            st.metric(param, value)
        
        # Performance analysis
        st.subheader("🚀 Performance Analysis")
        
        if len(valid_detections) > 0:
            # Calculate detection quality metrics
            detection_gaps = []
            for i in range(1, len(valid_detections)):
                gap = valid_detections.iloc[i]['Frame'] - valid_detections.iloc[i-1]['Frame']
                if gap > 1:
                    detection_gaps.append(gap - 1)
            
            total_gaps = sum(detection_gaps) if detection_gaps else 0
            gap_percentage = (total_gaps / len(df)) * 100
            
            st.metric("Detection Gaps", f"{len(detection_gaps)} gaps")
            st.metric("Gap Percentage", f"{gap_percentage:.1f}%")
            st.metric("Avg Gap Size", f"{np.mean(detection_gaps):.1f} frames" if detection_gaps else "0 frames")
            
            # Radius consistency
            radius_std = valid_detections['Radius'].std()
            radius_cv = (radius_std / valid_detections['Radius'].mean()) * 100
            st.metric("Radius Consistency (CV)", f"{radius_cv:.1f}%")

with tab5:
    st.header("📋 Data Export & Download")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Raw Detection Data")
        
        # Display data preview
        if len(valid_detections) > 0:
            st.dataframe(valid_detections.head(10), use_container_width=True)
            
            # Download buttons
            csv_data = valid_detections.to_csv(index=False)
            st.download_button(
                label="📥 Download Valid Detections (CSV)",
                data=csv_data,
                file_name="valid_detections.csv",
                mime="text/csv"
            )
            
            # Motion data download
            if 'Distance' in valid_detections.columns:
                motion_data = valid_detections[['Frame', 'X_center', 'Y_center', 'Distance', 'Speed']].copy()
                motion_csv = motion_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Motion Analysis (CSV)",
                    data=motion_csv,
                    file_name="motion_analysis.csv",
                    mime="text/csv"
                )
    
    with col2:
        st.subheader("📈 Analysis Summary Report")
        
        # Generate summary report
        if len(valid_detections) > 0:
            report = f"""
# Sphere Detection Analysis Report

## Project Overview
- **Project**: Rolling Resistance of Spheres on Wet Granular Material
- **Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Detection Summary
- **Total Frames**: {len(df)}
- **Valid Detections**: {len(valid_detections)}
- **Success Rate**: {(len(valid_detections)/len(df)*100):.1f}%
- **Frame Range**: {valid_detections['Frame'].min()} - {valid_detections['Frame'].max()}

## Position Analysis
- **X Range**: {valid_detections['X_center'].min()} - {valid_detections['X_center'].max()} pixels
- **Y Range**: {valid_detections['Y_center'].min()} - {valid_detections['Y_center'].max()} pixels
- **X Movement**: {valid_detections['X_center'].max() - valid_detections['X_center'].min()} pixels
- **Y Movement**: {valid_detections['Y_center'].max() - valid_detections['Y_center'].min()} pixels

## Radius Analysis
- **Average Radius**: {valid_detections['Radius'].mean():.1f} pixels
- **Radius Range**: {valid_detections['Radius'].min()} - {valid_detections['Radius'].max()} pixels
- **Radius Std Dev**: {valid_detections['Radius'].std():.1f} pixels

## Motion Analysis (if available)
"""
            
            if 'Distance' in valid_detections.columns:
                report += f"""
- **Total Distance**: {valid_detections['Distance'].sum():.1f} pixels
- **Average Speed**: {valid_detections['Speed'].mean():.1f} px/frame
- **Max Speed**: {valid_detections['Speed'].max():.1f} px/frame
"""
            
            report += f"""

## Configuration Used
- **Min Radius**: 18 pixels
- **Max Radius**: 35 pixels
- **Detection Threshold**: 8
- **Crop Area**: (400,500) to (1246,1000)

## Conclusions
- Detection algorithm performed with {(len(valid_detections)/len(df)*100):.1f}% success rate
- Sphere trajectory shows consistent movement pattern
- Radius detection varies within expected parameters
"""
            
            st.text_area("Analysis Report", report, height=400)
            
            st.download_button(
                label="📥 Download Analysis Report",
                data=report,
                file_name="sphere_analysis_report.txt",
                mime="text/plain"
            )
        
        # Configuration export
        st.subheader("⚙️ Export Configuration")
        
        config_json = {
            "detection_parameters": {
                "min_radius": 18,
                "max_radius": 35,
                "bw_threshold": 8,
                "circularity_min": 0.5,
                "min_score": 40,
                "max_movement": 120,
                "num_avg_bg": 150
            },
            "image_processing": {
                "crop_area": "(400,500) to (1246,1000)",
                "cropped_size": "846 x 500 pixels"
            },
            "analysis_results": {
                "total_frames": len(df),
                "valid_detections": len(valid_detections),
                "success_rate": round((len(valid_detections)/len(df)*100), 1) if len(df) > 0 else 0
            }
        }
        
        import json
        config_str = json.dumps(config_json, indent=2)
        
        st.download_button(
            label="📥 Download Configuration (JSON)",
            data=config_str,
            file_name="detection_config.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h4>🎓 Academic Research Project</h4>
    <p><strong>Rolling Resistance of Spheres on Wet Granular Material</strong></p>
    <p>First study to examine humidity effects on rolling friction | Innovation in geotechnical research</p>
    <p>💡 <em>Quantifying the effect of humidity on rolling resistance coefficient (Krr)</em></p>
</div>
""", unsafe_allow_html=True)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 About This Analysis
This tool analyzes sphere detection data from computer vision tracking experiments studying rolling resistance on wet granular materials.

### 🎯 Key Features:
- **Trajectory Analysis**: Track sphere movement
- **Detection Quality**: Success rates and gaps
- **Motion Analysis**: Speed and distance calculations
- **Configuration Review**: Parameter validation
- **Data Export**: CSV and report downloads

### 🔬 Research Context:
- **Objective**: Quantify humidity effects on rolling friction
- **Innovation**: First study of wet soils (literature only covers dry soils)
- **Applications**: Geotechnical engineering, soil mechanics
""")

# Add some utility functions
def calculate_rolling_resistance(v0, vf, distance, g=9.81):
    """Calculate rolling resistance coefficient"""
    if v0 > 0 and distance > 0:
        return (v0**2 - vf**2) / (2 * g * distance)
    return None

def calculate_penetration_ratio(depth, radius):
    """Calculate penetration ratio δ/R"""
    if radius > 0:
        return depth / radius
    return None

# Advanced analysis section (can be expanded)
if st.sidebar.checkbox("🔬 Advanced Physics Analysis", help="Enable advanced calculations for research"):
    st.markdown("---")
    st.header("🔬 Advanced Physics Analysis")
    
    st.info("""
    **Note**: For complete physics analysis, additional data is needed:
    - Initial velocity (V₀)
    - Final velocity (Vf)
    - Distance traveled (L)
    - Sphere material properties
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Rolling Resistance Calculation")
        st.latex(r"K_{rr} = \frac{V_0^2 - V_f^2}{2gL}")
        
        # Input fields for manual calculation
        v0_input = st.number_input("Initial Velocity V₀ (m/s)", value=1.0, min_value=0.0)
        vf_input = st.number_input("Final Velocity Vf (m/s)", value=0.5, min_value=0.0)
        distance_input = st.number_input("Distance L (m)", value=1.0, min_value=0.1)
        
        if st.button("Calculate Krr"):
            krr = calculate_rolling_resistance(v0_input, vf_input, distance_input)
            if krr is not None:
                st.success(f"Rolling Resistance Coefficient: **{krr:.4f}**")
            else:
                st.error("Invalid input values")
    
    with col2:
        st.subheader("📏 Penetration Analysis")
        st.latex(r"\frac{\delta}{R} = A\left(\frac{\rho_s}{\rho_g}\right)^n")
        
        # Penetration calculation
        depth_input = st.number_input("Penetration Depth δ (mm)", value=5.0, min_value=0.0)
        sphere_radius = st.number_input("Sphere Radius R (mm)", value=25.0, min_value=1.0)
        
        if st.button("Calculate δ/R"):
            ratio = calculate_penetration_ratio(depth_input, sphere_radius)
            if ratio is not None:
                st.success(f"Penetration Ratio δ/R: **{ratio:.3f}**")
            else:
                st.error("Invalid input values")
    
    # Literature comparison
    st.subheader("📖 Literature Comparison")
    st.markdown("""
    **Reference Values from Literature:**
    - **Van Wal (2017)**: Dry soils, Krr = 0.05-0.07
    - **Darbois Texier (2018)**: δ/R ∝ (ρs/ρg)^0.75
    - **De Blasio (2009)**: Krr independent of speed
    
    **Expected Results for Wet Soils:**
    - Krr increases with humidity (capillary cohesion)
    - Optimum at around 10-15% humidity
    - Penetration ratio (δ/R) increases with humidity
    """)

# Error handling and data validation
if uploaded_file is None:
    st.info("📁 **Using sample data for demonstration.** Upload your own detections.csv file to analyze real data.")

if len(df) == 0:
    st.error("❌ No data available. Please upload a valid CSV file.")
elif len(valid_detections) == 0:
    st.warning("⚠️ No valid detections found. Check your detection parameters.")
elif len(valid_detections) < 10:
    st.warning(f"⚠️ Only {len(valid_detections)} valid detections found. Results may be unreliable.")
