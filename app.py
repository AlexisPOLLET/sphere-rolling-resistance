# === CODE 3 : ANALYSE AVANCÉE ET COMPLÈTE ===
    elif analysis_type == "🔬 Code 3 : Analyse Complète":
        st.markdown("""
        <div class="section-header">
            <h2>🔬 Code 3 : Analyse Cinématique Avancée et Complète</h2>
            <p>Analyse approfondie avec debug et métriques avancées</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Vérification des données
        st.markdown("### 🔍 Vérification des Données")
        if len(df_valid) > 0:
            colimport streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Configuration de la page
st.set_page_config(
    page_title="Sphere Rolling Resistance Analysis",
    page_icon="⚪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
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
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("""
# 🎾 Sphere Rolling Resistance Analysis Platform
## 🔬 Complete Analysis Suite for Granular Mechanics Research
*Upload your data and access our 3 specialized analysis tools*
""")

# Section d'upload de fichier
st.markdown("""
<div class="upload-section">
    <h2>📂 Upload Your Experimental Data</h2>
    <p>Start by uploading your CSV file with detection results to get a personalized analysis</p>
</div>
""", unsafe_allow_html=True)

# Upload de fichier
uploaded_file = st.file_uploader(
    "Choose your CSV file with detection data", 
    type=['csv'],
    help="Upload a CSV file with columns: Frame, X_center, Y_center, Radius"
)

# Variables globales pour les données
df = None
df_valid = None

# Fonction pour charger les données
@st.cache_data
def load_uploaded_data(uploaded_file):
    """Charge les données depuis le fichier uploadé"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Vérifier les colonnes requises
        required_columns = ['Frame', 'X_center', 'Y_center', 'Radius']
        if not all(col in df.columns for col in required_columns):
            st.error(f"❌ Le fichier doit contenir les colonnes : {required_columns}")
            st.error(f"📊 Colonnes trouvées : {list(df.columns)}")
            return None, None
        
        # Filtrer les détections valides
        df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
        
        return df, df_valid
    return None, None

# Fonction pour créer des données d'exemple
def create_sample_data():
    """Crée des données d'exemple pour la démonstration"""
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

# Charger les données
if uploaded_file is not None:
    df, df_valid = load_uploaded_data(uploaded_file)
    if df is not None:
        st.success(f"✅ Fichier chargé avec succès ! {len(df)} frames détectées")
else:
    # Option pour utiliser des données d'exemple
    if st.button("🔬 Utiliser des données d'exemple pour la démonstration"):
        df = create_sample_data()
        df_valid = df[(df['X_center'] != 0) & (df['Y_center'] != 0) & (df['Radius'] != 0)]
        st.info("📊 Données d'exemple chargées - vous pouvez maintenant explorer les fonctionnalités")

# Navigation seulement si des données sont chargées
if df is not None:
    
    # Aperçu rapide des données
    st.markdown("### 📊 Aperçu de vos données")
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
            <p>Détections Valides</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        success_rate = len(df_valid) / len(df) * 100 if len(df) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{success_rate:.1f}%</h3>
            <p>Taux de Succès</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        avg_radius = df_valid['Radius'].mean() if len(df_valid) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_radius:.1f} px</h3>
            <p>Rayon Moyen</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation entre les 3 codes
    st.markdown("---")
    st.markdown("## 🔧 Choisissez votre analyse")
    
    # Sidebar pour navigation
    st.sidebar.title("🧭 Navigation")
    analysis_type = st.sidebar.selectbox("Sélectionnez le type d'analyse :", [
        "📈 Code 1 : Visualisation de Trajectoire",
        "📊 Code 2 : Analyse Krr",
        "🔬 Code 3 : Analyse Complète",
        "📋 Vue d'ensemble des données"
    ])
    
    # === CODE 1 : DETECTION ET VISUALISATION DE TRAJECTOIRE ===
    if analysis_type == "📈 Code 1 : Visualisation de Trajectoire":
        st.markdown("""
        <div class="section-header">
            <h2>📈 Code 1 : Détection et Visualisation de Trajectoire</h2>
            <p>Système complet de détection de sphères avec analyse de trajectoire</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Configuration de détection
        st.markdown("### ⚙️ Configuration de Détection")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Paramètres de Taille**")
            minR = st.slider("Rayon minimum", 10, 30, 18)
            maxR = st.slider("Rayon maximum", 25, 50, 35)
            
        with col2:
            st.markdown("**Paramètres de Détection**")
            bw_threshold = st.slider("Seuil de détection", 1, 20, 8)
            min_score = st.slider("Score minimum", 20, 60, 40)
            
        with col3:
            st.markdown("**Paramètres de Forme**")
            circularity_min = st.slider("Circularité minimum", 0.1, 1.0, 0.5)
            max_movement = st.slider("Mouvement max", 50, 200, 120)
        
        # Visualisation des données chargées
        if len(df_valid) > 0:
            st.markdown("### 🎯 Trajectoire de la Sphère Détectée")
            
            # Graphique principal de trajectoire
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('🛤️ Trajectoire Complète', '📍 Position X vs Temps', 
                               '📍 Position Y vs Temps', '⚪ Évolution du Rayon'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Trajectoire avec gradient de couleur basé sur le temps
            fig.add_trace(
                go.Scatter(x=df_valid['X_center'], y=df_valid['Y_center'],
                          mode='markers+lines', 
                          marker=dict(color=df_valid['Frame'], 
                                    colorscale='viridis', 
                                    size=8,
                                    colorbar=dict(title="Frame")),
                          line=dict(width=2),
                          name='Trajectoire'),
                row=1, col=1
            )
            
            # Position X
            fig.add_trace(
                go.Scatter(x=df_valid['Frame'], y=df_valid['X_center'],
                          mode='lines+markers', 
                          line=dict(color='#3498db', width=3),
                          name='Position X'),
                row=1, col=2
            )
            
            # Position Y
            fig.add_trace(
                go.Scatter(x=df_valid['Frame'], y=df_valid['Y_center'],
                          mode='lines+markers',
                          line=dict(color='#e74c3c', width=3),
                          name='Position Y'),
                row=2, col=1
            )
            
            # Rayon détecté
            fig.add_trace(
                go.Scatter(x=df_valid['Frame'], y=df_valid['Radius'],
                          mode='lines+markers',
                          line=dict(color='#2ecc71', width=3),
                          name='Rayon'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=False,
                             title_text="Analyse Complète de Détection")
            
            # Inverser l'axe Y pour la trajectoire (coordonnées image)
            fig.update_yaxes(autorange="reversed", row=1, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques de détection
            st.markdown("### 📊 Statistiques de Détection")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_distance = np.sqrt(
                    (df_valid['X_center'].iloc[-1] - df_valid['X_center'].iloc[0])**2 + 
                    (df_valid['Y_center'].iloc[-1] - df_valid['Y_center'].iloc[0])**2
                )
                st.metric("Distance Totale", f"{total_distance:.1f} px")
                
            with col2:
                if len(df_valid) > 1:
                    dx = df_valid['X_center'].diff()
                    dy = df_valid['Y_center'].diff()
                    speed = np.sqrt(dx**2 + dy**2)
                    avg_speed = speed.mean()
                    st.metric("Vitesse Moyenne", f"{avg_speed:.2f} px/frame")
                    
            with col3:
                vertical_displacement = abs(df_valid['Y_center'].iloc[-1] - df_valid['Y_center'].iloc[0])
                st.metric("Déplacement Vertical", f"{vertical_displacement:.1f} px")
                
            with col4:
                avg_radius = df_valid['Radius'].mean()
                radius_std = df_valid['Radius'].std()
                st.metric("Rayon Moyen", f"{avg_radius:.1f} ± {radius_std:.1f} px")
            
            # Analyse de qualité de détection
            st.markdown("### 🔍 Qualité de Détection")
            
            # Graphique de distribution des rayons
            col1, col2 = st.columns(2)
            
            with col1:
                fig_radius = px.histogram(df_valid, x='Radius', nbins=15,
                                         title="Distribution des Rayons Détectés",
                                         labels={'Radius': 'Rayon (pixels)', 'count': 'Fréquence'})
                fig_radius.add_vline(x=minR, line_dash="dash", line_color="red", 
                                    annotation_text=f"Min: {minR}")
                fig_radius.add_vline(x=maxR, line_dash="dash", line_color="red", 
                                    annotation_text=f"Max: {maxR}")
                st.plotly_chart(fig_radius, use_container_width=True)
                
            with col2:
                # Analyse de continuité du mouvement
                if len(df_valid) > 1:
                    movement = np.sqrt(dx**2 + dy**2)
                    fig_movement = px.line(x=df_valid['Frame'][1:], y=movement,
                                          title="Mouvement Inter-Frame",
                                          labels={'x': 'Frame', 'y': 'Déplacement (pixels)'})
                    fig_movement.add_hline(y=max_movement, line_dash="dash", line_color="red",
                                          annotation_text=f"Max autorisé: {max_movement}")
                    st.plotly_chart(fig_movement, use_container_width=True)
        
        # Information sur l'algorithme de détection
        st.markdown("### 🧠 Algorithme de Détection")
        st.markdown("""
        **Méthode utilisée :** Détection de cercles par soustraction de fond
        
        **Étapes principales :**
        1. **Création du fond** : Moyenne de 150 images de référence
        2. **Soustraction** : Élimination du fond statique
        3. **Seuillage** : Binarisation avec seuil adaptatif
        4. **Morphologie** : Nettoyage des contours
        5. **Détection** : Recherche de contours circulaires
        6. **Validation** : Filtrage par taille, forme et continuité
        
        **Critères de qualité :**
        - Taille : {minR} ≤ rayon ≤ {maxR} pixels
        - Forme : Circularité ≥ {circularity_min}
        - Continuité : Mouvement ≤ {max_movement} pixels/frame
        - Score : Qualité globale ≥ {min_score}
        """.format(minR=minR, maxR=maxR, circularity_min=circularity_min, 
                  max_movement=max_movement, min_score=min_score))
    
    # === CODE 2 : ANALYSE KRR ===
    elif analysis_type == "📊 Code 2 : Analyse Krr":
        st.markdown("""
        <div class="section-header">
            <h2>📊 Code 2 : Analyse du Coefficient de Résistance au Roulement (Krr)</h2>
            <p>Calculs physiques complets pour déterminer le coefficient Krr</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Paramètres de la sphère
        st.markdown("### 🔵 Paramètres de la Sphère")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sphere_radius_mm = st.number_input("Rayon de la sphère (mm)", value=15.0, min_value=1.0, max_value=50.0)
            sphere_mass_g = st.number_input("Masse de la sphère (g)", value=10.0, min_value=0.1, max_value=1000.0)
            
        with col2:
            sphere_type = st.selectbox("Type de sphère", ["Solide (j=2/5)", "Creuse (j=2/3)"])
            j_value = 2/5 if "Solide" in sphere_type else 2/3
            
            # Calcul de densité
            volume_mm3 = (4/3) * np.pi * sphere_radius_mm**3
            volume_m3 = volume_mm3 * 1e-9
            mass_kg = sphere_mass_g * 1e-3
            density_kg_m3 = mass_kg / volume_m3
            st.metric("Densité", f"{density_kg_m3:.0f} kg/m³")
            
        with col3:
            st.metric("Facteur d'inertie j", f"{j_value:.3f}")
            st.metric("Facteur (1+j)⁻¹", f"{1/(1+j_value):.4f}")
        
        # Paramètres expérimentaux
        st.markdown("### 📐 Paramètres Expérimentaux")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fps = st.number_input("FPS de la caméra", value=250.0, min_value=1.0, max_value=1000.0)
            angle_deg = st.number_input("Angle d'inclinaison (°)", value=15.0, min_value=0.1, max_value=45.0)
            
        with col2:
            # Calibration automatique basée sur le rayon détecté
            if len(df_valid) > 0:
                avg_radius_pixels = df_valid['Radius'].mean()
                auto_calibration = avg_radius_pixels / sphere_radius_mm
                st.metric("Calibration auto", f"{auto_calibration:.2f} px/mm")
                
                use_auto_cal = st.checkbox("Utiliser calibration automatique", value=True)
                if use_auto_cal:
                    pixels_per_mm = auto_calibration
                else:
                    pixels_per_mm = st.number_input("Calibration (px/mm)", value=auto_calibration, min_value=0.1)
            else:
                pixels_per_mm = st.number_input("Calibration (px/mm)", value=5.0, min_value=0.1)
                
        with col3:
            water_content = st.number_input("Teneur en eau (%)", value=0.0, min_value=0.0, max_value=100.0)
            
        # Calculs cinématiques et Krr
        if len(df_valid) > 10:
            st.markdown("### 🧮 Calculs Cinématiques")
            
            # Conversion des unités
            dt = 1 / fps  # s
            
            # Positions en mètres
            x_mm = df_valid['X_center'].values / pixels_per_mm
            y_mm = df_valid['Y_center'].values / pixels_per_mm
            x_m = x_mm / 1000
            y_m = y_mm / 1000
            
            # Temps
            t = np.arange(len(df_valid)) * dt
            
            # Vitesses
            vx = np.gradient(x_m, dt)
            vy = np.gradient(y_m, dt)
            v_magnitude = np.sqrt(vx**2 + vy**2)
            
            # Vitesses initiale et finale (moyenne sur quelques points)
            n_avg = min(3, len(v_magnitude)//4)
            v0 = np.mean(v_magnitude[:n_avg])
            vf = np.mean(v_magnitude[-n_avg:])
            
            # Distance totale
            distances = np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2)
            total_distance = np.sum(distances)
            
            # Calcul du coefficient Krr
            g = 9.81  # m/s²
            if total_distance > 0:
                krr = (v0**2 - vf**2) / (2 * g * total_distance)
                
                # Coefficient de friction effectif
                angle_rad = np.radians(angle_deg)
                mu_eff = krr + np.tan(angle_rad)
                
                # Affichage des résultats
                st.markdown("### 📈 Résultats Krr")
                
                result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                
                with result_col1:
                    st.metric("V₀ (vitesse initiale)", f"{v0*1000:.1f} mm/s")
                    st.caption(f"{v0:.4f} m/s")
                    
                with result_col2:
                    st.metric("Vf (vitesse finale)", f"{vf*1000:.1f} mm/s") 
                    st.caption(f"{vf:.4f} m/s")
                    
                with result_col3:
                    st.metric("Distance totale", f"{total_distance*1000:.1f} mm")
                    st.caption(f"{total_distance:.4f} m")
                    
                with result_col4:
                    st.metric("**Coefficient Krr**", f"{krr:.6f}")
                    if 0.03 <= krr <= 0.10:
                        st.success("✅ Cohérent avec Van Wal (2017)")
                    elif krr < 0:
                        st.error("⚠️ Krr négatif - sphère accélère")
                    else:
                        st.warning("⚠️ Différent de la littérature")
                
                # Graphiques cinématiques
                st.markdown("### 📊 Analyse Cinématique")
                
                fig_kinematics = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Vitesse vs Temps', 'Accélération vs Temps', 
                                   'Trajectoire', 'Composantes de Vitesse')
                )
                
                # Vitesse
                fig_kinematics.add_trace(
                    go.Scatter(x=t, y=v_magnitude*1000, mode='lines', name='Vitesse',
                              line=dict(color='blue', width=2)),
                    row=1, col=1
                )
                fig_kinematics.add_hline(y=v0*1000, line_dash="dash", line_color="green", row=1, col=1)
                fig_kinematics.add_hline(y=vf*1000, line_dash="dash", line_color="red", row=1, col=1)
                
                # Accélération
                acceleration = np.gradient(v_magnitude, dt)
                fig_kinematics.add_trace(
                    go.Scatter(x=t, y=acceleration*1000, mode='lines', name='Accélération',
                              line=dict(color='red', width=2)),
                    row=1, col=2
                )
                
                # Trajectoire
                fig_kinematics.add_trace(
                    go.Scatter(x=x_mm, y=y_mm, mode='markers+lines', name='Trajectoire',
                              marker=dict(color=t, colorscale='viridis')),
                    row=2, col=1
                )
                
                # Composantes de vitesse
                fig_kinematics.add_trace(
                    go.Scatter(x=t, y=np.abs(vx)*1000, mode='lines', name='|Vx|',
                              line=dict(color='blue', width=2)),
                    row=2, col=2
                )
                fig_kinematics.add_trace(
                    go.Scatter(x=t, y=vy*1000, mode='lines', name='Vy',
                              line=dict(color='red', width=2)),
                    row=2, col=2
                )
                
                fig_kinematics.update_layout(height=800, showlegend=False)
                fig_kinematics.update_xaxes(title_text="Temps (s)", row=1)
                fig_kinematics.update_xaxes(title_text="Temps (s)", row=2)
                fig_kinematics.update_yaxes(title_text="Vitesse (mm/s)", row=1, col=1)
                fig_kinematics.update_yaxes(title_text="Accél. (mm/s²)", row=1, col=2)
                fig_kinematics.update_yaxes(title_text="Y (mm)", row=2, col=1)
                fig_kinematics.update_yaxes(title_text="Vitesse (mm/s)", row=2, col=2)
                
                st.plotly_chart(fig_kinematics, use_container_width=True)
                
                # Analyse énergétique
                st.markdown("### ⚡ Analyse Énergétique")
                
                # Énergies
                E_trans = 0.5 * mass_kg * v_magnitude**2
                I = j_value * mass_kg * (sphere_radius_mm/1000)**2
                omega = v_magnitude / (sphere_radius_mm/1000)
                E_rot = 0.5 * I * omega**2
                E_total = E_trans + E_rot
                
                energy_col1, energy_col2, energy_col3 = st.columns(3)
                
                with energy_col1:
                    st.metric("Énergie initiale", f"{E_total[0]*1000:.2f} mJ")
                    
                with energy_col2:
                    st.metric("Énergie finale", f"{E_total[-1]*1000:.2f} mJ")
                    
                with energy_col3:
                    energy_dissipated = (E_total[0] - E_total[-1]) * 1000
                    st.metric("Énergie dissipée", f"{energy_dissipated:.2f} mJ")
                
                # Graphique énergétique
                fig_energy = go.Figure()
                fig_energy.add_trace(go.Scatter(x=t, y=E_trans*1000, mode='lines', name='Translation', line=dict(color='blue')))
                fig_energy.add_trace(go.Scatter(x=t, y=E_rot*1000, mode='lines', name='Rotation', line=dict(color='red')))
                fig_energy.add_trace(go.Scatter(x=t, y=E_total*1000, mode='lines', name='Total', line=dict(color='black', width=3)))
                
                fig_energy.update_layout(
                    title="Évolution des Énergies Cinétiques",
                    xaxis_title="Temps (s)",
                    yaxis_title="Énergie (mJ)",
                    height=400
                )
                
                st.plotly_chart(fig_energy, use_container_width=True)
                
                # Comparaison avec la littérature
                st.markdown("### 📚 Comparaison avec la Littérature")
                
                literature_krr = [0.05, 0.07]  # Van Wal (2017)
                
                comparison_data = {
                    'Source': ['Van Wal (2017) - Min', 'Van Wal (2017) - Max', 'Expérience Actuelle'],
                    'Krr': [0.05, 0.07, krr],
                    'Conditions': ['Sol sec', 'Sol sec', f'w = {water_content}%']
                }
                
                fig_comparison = px.bar(comparison_data, x='Source', y='Krr', color='Conditions',
                                       title="Comparaison des Coefficients Krr")
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Sauvegarde des résultats
                results_summary = {
                    'sphere_radius_mm': sphere_radius_mm,
                    'sphere_mass_g': sphere_mass_g,
                    'sphere_type': sphere_type,
                    'density_kg_m3': density_kg_m3,
                    'angle_deg': angle_deg,
                    'water_content': water_content,
                    'v0_ms': v0,
                    'vf_ms': vf,
                    'distance_m': total_distance,
                    'krr': krr,
                    'mu_eff': mu_eff,
                    'energy_dissipated_mJ': energy_dissipated
                }
                
                st.markdown("### 💾 Résumé des Résultats")
                st.json(results_summary)
                
            else:
                st.error("❌ Distance parcourue nulle - impossible de calculer Krr")
        else:
            st.warning("⚠️ Pas assez de données valides pour l'analyse Krr")
    
    # === CODE 3 : ANALYSE AVANCÉE ET COMPLÈTE ===
    elif analysis_type == "🔬 Code 3 : Analyse Complète":
        st.markdown("""
        <div class="section-header">
            <h2>🔬 Code 3 : Analyse Cinématique Avancée et Complète</h2>
            <p>Analyse approfondie avec debug et métriques avancées</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Vérification des données
        st.markdown("### 🔍 Vérification des Données")
        if len(df_valid) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Données valides", f"{len(df_valid)} frames")
                st.metric("Taux de succès", f"{len(df_valid)/len(df)*100:.1f}%")
                
            with col2:
                velocity_range = df_valid['Radius'].max() - df_valid['Radius'].min()
                st.metric("Variation de rayon", f"{velocity_range:.1f} px")
                st.metric("Première détection", f"Frame {df_valid['Frame'].min()}")
                
            with col3:
                st.metric("Dernière détection", f"Frame {df_valid['Frame'].max()}")
                duration_frames = df_valid['Frame'].max() - df_valid['Frame'].min()
                st.metric("Durée de suivi", f"{duration_frames} frames")
            
            # Paramètres pour l'analyse avancée
            st.markdown("### ⚙️ Paramètres d'Analyse Avancée")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Paramètres Sphère**")
                mass_g = st.number_input("Masse (g)", value=10.0, min_value=0.1, key="adv_mass")
                radius_mm = st.number_input("Rayon (mm)", value=15.0, min_value=1.0, key="adv_radius")
                sphere_type = st.selectbox("Type", ["Solide", "Creuse"], key="adv_type")
                j_factor = 2/5 if sphere_type == "Solide" else 2/3
                
            with col2:
                st.markdown("**Paramètres Expérimentaux**")
                fps = st.number_input("FPS", value=250.0, min_value=1.0, key="adv_fps")
                angle_deg = st.number_input("Angle (°)", value=15.0, min_value=0.1, key="adv_angle")
                
                # Calibration automatique
                if len(df_valid) > 0:
                    avg_radius_px = df_valid['Radius'].mean()
                    auto_cal = avg_radius_px / radius_mm
                    st.metric("Calibration auto", f"{auto_cal:.2f} px/mm")
                    pixels_per_mm = auto_cal
                
            with col3:
                st.markdown("**Filtrage des Données**")
                use_smoothing = st.checkbox("Lissage des données", value=True)
                smooth_window = st.slider("Fenêtre de lissage", 3, 11, 5, step=2)
                remove_outliers = st.checkbox("Supprimer les aberrants", value=True)
                
            # Calculs cinématiques avancés
            if st.button("🚀 Lancer l'Analyse Complète"):
                
                st.markdown("### 🧮 Calculs Cinématiques Avancés")
                
                # Extraction et préparation des données
                t = np.arange(len(df_valid)) / fps
                x_mm = df_valid['X_center'].values / pixels_per_mm
                y_mm = df_valid['Y_center'].values / pixels_per_mm
                x_m = x_mm / 1000
                y_m = y_mm / 1000
                
                # Suppression des aberrants si demandé
                if remove_outliers:
                    # Détection simple des aberrants par écart-type
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
                    
                    st.info(f"🔧 Aberrants supprimés : {np.sum(~mask)} points")
                
                # Calcul des vitesses avec lissage optionnel
                dt = np.mean(np.diff(t))
                
                if use_smoothing and len(x_m) >= smooth_window:
                    from scipy.signal import savgol_filter
                    try:
                        x_smooth = savgol_filter(x_m, smooth_window, 2)
                        y_smooth = savgol_filter(y_m, smooth_window, 2)
                        vx = np.gradient(x_smooth, dt)
                        vy = np.gradient(y_smooth, dt)
                        st.success(f"✅ Données lissées avec fenêtre {smooth_window}")
                    except:
                        vx = np.gradient(x_m, dt)
                        vy = np.gradient(y_m, dt)
                        st.warning("⚠️ Lissage échoué, utilisation des données brutes")
                else:
                    vx = np.gradient(x_m, dt)
                    vy = np.gradient(y_m, dt)
                
                v_magnitude = np.sqrt(vx**2 + vy**2)
                
                # Accélération
                acceleration = np.gradient(v_magnitude, dt)
                
                # Paramètres physiques
                mass_kg = mass_g / 1000
                radius_m = radius_mm / 1000
                angle_rad = np.radians(angle_deg)
                g = 9.81
                
                # Forces et énergies
                F_resistance = mass_kg * acceleration
                F_gravity = mass_kg * g * np.sin(angle_rad)
                
                # Énergies cinétiques
                E_trans = 0.5 * mass_kg * v_magnitude**2
                I = j_factor * mass_kg * radius_m**2
                omega = v_magnitude / radius_m
                E_rot = 0.5 * I * omega**2
                E_total = E_trans + E_rot
                
                # Puissance et Krr instantané
                P_resistance = np.abs(F_resistance * v_magnitude)
                Krr_inst = np.abs(F_resistance) / (mass_kg * g)
                
                # Métriques globales
                avg_krr = np.mean(Krr_inst)
                energy_dissipated = (E_total[0] - E_total[-1]) * 1000  # mJ
                avg_power = np.mean(P_resistance) * 1000  # mW
                
                # Affichage des résultats
                st.markdown("### 📊 Résultats de l'Analyse Avancée")
                
                result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                
                with result_col1:
                    st.metric("Krr Moyen", f"{avg_krr:.6f}")
                    st.metric("Krr Médian", f"{np.median(Krr_inst):.6f}")
                    
                with result_col2:
                    st.metric("Énergie Dissipée", f"{energy_dissipated:.2f} mJ")
                    st.metric("Puissance Moyenne", f"{avg_power:.2f} mW")
                    
                with result_col3:
                    st.metric("Vitesse Max", f"{np.max(v_magnitude)*1000:.1f} mm/s")
                    st.metric("Vitesse Min", f"{np.min(v_magnitude)*1000:.1f} mm/s")
                    
                with result_col4:
                    st.metric("Accél. Max", f"{np.max(np.abs(acceleration))*1000:.1f} mm/s²")
                    total_distance = np.sum(np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2))
                    st.metric("Distance Totale", f"{total_distance*1000:.1f} mm")
                
                # Graphiques avancés
                st.markdown("### 📈 Visualisations Avancées")
                
                # Figure principale avec 6 sous-graphiques
                fig_advanced = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=('Vitesse Lissée vs Temps', 'Accélération vs Temps',
                                   'Énergies Cinétiques', 'Krr Instantané',
                                   'Puissance de Résistance', 'Forces'),
                    vertical_spacing=0.08
                )
                
                # 1. Vitesse
                fig_advanced.add_trace(
                    go.Scatter(x=t, y=v_magnitude*1000, mode='lines', name='Vitesse',
                              line=dict(color='blue', width=2)),
                    row=1, col=1
                )
                
                # 2. Accélération
                fig_advanced.add_trace(
                    go.Scatter(x=t, y=acceleration*1000, mode='lines', name='Accélération',
                              line=dict(color='red', width=2)),
                    row=1, col=2
                )
                
                # 3. Énergies
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
                
                # 4. Krr instantané
                fig_advanced.add_trace(
                    go.Scatter(x=t, y=Krr_inst, mode='lines', name='Krr',
                              line=dict(color='purple', width=2)),
                    row=2, col=2
                )
                fig_advanced.add_hline(y=avg_krr, line_dash="dash", line_color="orange", row=2, col=2)
                
                # 5. Puissance
                fig_advanced.add_trace(
                    go.Scatter(x=t, y=P_resistance*1000, mode='lines', name='Puissance',
                              line=dict(color='green', width=2)),
                    row=3, col=1
                )
                
                # 6. Forces
                fig_advanced.add_trace(
                    go.Scatter(x=t, y=F_resistance*1000, mode='lines', name='F_résistance',
                              line=dict(color='red', width=2)),
                    row=3, col=2
                )
                fig_advanced.add_hline(y=F_gravity*1000, line_dash="dash", line_color="blue", row=3, col=2)
                
                # Mise à jour des axes
                fig_advanced.update_xaxes(title_text="Temps (s)")
                fig_advanced.update_yaxes(title_text="Vitesse (mm/s)", row=1, col=1)
                fig_advanced.update_yaxes(title_text="Accélération (mm/s²)", row=1, col=2)
                fig_advanced.update_yaxes(title_text="Énergie (mJ)", row=2, col=1)
                fig_advanced.update_yaxes(title_text="Coefficient Krr", row=2, col=2)
                fig_advanced.update_yaxes(title_text="Puissance (mW)", row=3, col=1)
                fig_advanced.update_yaxes(title_text="Force (mN)", row=3, col=2)
                
                fig_advanced.update_layout(height=900, showlegend=False)
                st.plotly_chart(fig_advanced, use_container_width=True)
                
                # Analyse statistique détaillée
                st.markdown("### 📊 Analyse Statistique Détaillée")
                
                stats_data = {
                    'Paramètre': ['Krr', 'Vitesse (mm/s)', 'Accélération (mm/s²)', 
                                 'Puissance (mW)', 'Force résistance (mN)'],
                    'Moyenne': [f"{np.mean(Krr_inst):.6f}", f"{np.mean(v_magnitude)*1000:.2f}",
                               f"{np.mean(acceleration)*1000:.2f}", f"{np.mean(P_resistance)*1000:.2f}",
                               f"{np.mean(F_resistance)*1000:.2f}"],
                    'Écart-type': [f"{np.std(Krr_inst):.6f}", f"{np.std(v_magnitude)*1000:.2f}",
                                  f"{np.std(acceleration)*1000:.2f}", f"{np.std(P_resistance)*1000:.2f}",
                                  f"{np.std(F_resistance)*1000:.2f}"],
                    'Min': [f"{np.min(Krr_inst):.6f}", f"{np.min(v_magnitude)*1000:.2f}",
                           f"{np.min(acceleration)*1000:.2f}", f"{np.min(P_resistance)*1000:.2f}",
                           f"{np.min(F_resistance)*1000:.2f}"],
                    'Max': [f"{np.max(Krr_inst):.6f}", f"{np.max(v_magnitude)*1000:.2f}",
                           f"{np.max(acceleration)*1000:.2f}", f"{np.max(P_resistance)*1000:.2f}",
                           f"{np.max(F_resistance)*1000:.2f}"]
                }
                
                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats, use_container_width=True)
                
                # Comparaison avec la littérature et analyse de cohérence
                st.markdown("### 📚 Analyse de Cohérence")
                
                coherence_col1, coherence_col2 = st.columns(2)
                
                with coherence_col1:
                    st.markdown("**Cohérence avec Van Wal (2017)**")
                    if 0.03 <= avg_krr <= 0.10:
                        st.success(f"✅ Krr = {avg_krr:.6f} cohérent avec littérature (0.05-0.07)")
                    elif avg_krr < 0:
                        st.error(f"❌ Krr négatif = {avg_krr:.6f} - Sphère accélère")
                    else:
                        st.warning(f"⚠️ Krr = {avg_krr:.6f} différent de la littérature")
                    
                    # Test d'ordre de grandeur
                    if 0.001 <= abs(avg_krr) <= 1.0:
                        st.success("✅ Ordre de grandeur correct")
                    else:
                        st.error("❌ Ordre de grandeur aberrant")
                
                with coherence_col2:
                    st.markdown("**Bilan Énergétique**")
                    energy_ratio = energy_dissipated / (E_total[0] * 1000) * 100
                    st.metric("Énergie dissipée", f"{energy_ratio:.1f}%")
                    
                    if 10 <= energy_ratio <= 90:
                        st.success("✅ Dissipation énergétique cohérente")
                    else:
                        st.warning("⚠️ Dissipation énergétique inhabituelle")
                
                # Sauvegarde des résultats avancés
                st.markdown("### 💾 Export des Résultats")
                
                advanced_results = {
                    'configuration': {
                        'mass_g': mass_g,
                        'radius_mm': radius_mm,
                        'sphere_type': sphere_type,
                        'j_factor': j_factor,
                        'angle_deg': angle_deg,
                        'fps': fps,
                        'pixels_per_mm': pixels_per_mm,
                        'smoothing': use_smoothing,
                        'outlier_removal': remove_outliers
                    },
                    'results': {
                        'avg_krr': float(avg_krr),
                        'krr_std': float(np.std(Krr_inst)),
                        'energy_dissipated_mJ': float(energy_dissipated),
                        'avg_power_mW': float(avg_power),
                        'total_distance_mm': float(total_distance * 1000),
                        'max_velocity_mm_s': float(np.max(v_magnitude) * 1000),
                        'max_acceleration_mm_s2': float(np.max(np.abs(acceleration)) * 1000)
                    },
                    'quality_metrics': {
                        'data_points': len(t),
                        'tracking_duration_s': float(t[-1] - t[0]),
                        'coherence_with_literature': 0.03 <= avg_krr <= 0.10,
                        'energy_conservation': abs(energy_ratio) < 100
                    }
                }
                
                st.json(advanced_results)
                
                # Bouton de téléchargement des données détaillées
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
                    label="📥 Télécharger données détaillées (CSV)",
                    data=csv_data,
                    file_name="analyse_cinetique_avancee.csv",
                    mime="text/csv"
                )
                
        else:
            st.error("❌ Aucune donnée valide pour l'analyse avancée")
    
    # === VUE D'ENSEMBLE ===
    else:  # Vue d'ensemble des données
        st.markdown("""
        <div class="section-header">
            <h2>📋 Vue d'ensemble de vos données</h2>
            <p>Exploration et validation de la qualité des données</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Affichage des premières lignes
        st.markdown("### 📊 Aperçu des Données")
        st.dataframe(df.head(10))
        
        # Statistiques descriptives
        st.markdown("### 📈 Statistiques Descriptives")
        st.dataframe(df_valid.describe())
        
        # Graphique de distribution
        if len(df_valid) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.histogram(df_valid, x='Radius', title="Distribution des Rayons")
                st.plotly_chart(fig1, use_container_width=True)
                
            with col2:
                detection_status = df['X_center'] != 0
                fig2 = px.pie(values=[detection_status.sum(), (~detection_status).sum()],
                             names=['Détecté', 'Non détecté'],
                             title="Répartition des Détections")
                st.plotly_chart(fig2, use_container_width=True)

else:
    # Message si aucune donnée n'est chargée
    st.markdown("""
    ## 🚀 Pour commencer :
    
    1. **📂 Uploadez votre fichier CSV** avec vos données expérimentales
    2. **Ou cliquez sur "Utiliser des données d'exemple"** pour explorer les fonctionnalités
    3. **🔧 Choisissez l'analyse** qui vous intéresse dans le menu
    
    ### 📋 Format de fichier attendu :
    Votre CSV doit contenir les colonnes suivantes :
    - `Frame` : Numéro de l'image
    - `X_center` : Position X du centre de la sphère
    - `Y_center` : Position Y du centre de la sphère  
    - `Radius` : Rayon détecté de la sphère
    
    ### 🔧 Les 3 Codes Intégrés :
    - **Code 1** : Visualisation de trajectoire
    - **Code 2** : Analyse Krr (coefficient de résistance)
    - **Code 3** : Analyse complète et avancée
    """)

# Footer
st.markdown("---")
st.markdown("""
### 🎓 Sphere Rolling Resistance Analysis Platform
*Développé pour l'analyse de la résistance au roulement de sphères sur matériau granulaire humide*
""")
