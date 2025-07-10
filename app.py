# =============================================================================
# ANALYSE CINÉMATIQUE AVANCÉE
# Calculs d'énergie, forces, puissance et validation théorique
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class AdvancedKinematicsAnalyzer:
    def __init__(self):
        self.g = 9.81  # m/s²
        self.data = None
        self.sphere_params = {}
        self.experimental_params = {}
        self.kinematics = {}
        self.advanced_results = {}
        
    def load_existing_analysis(self, csv_file="results/kinematics_analysis.csv", 
                               config_file="results/rolling_resistance_analysis.txt"):
        """Charger les données d'une analyse précédente"""
        try:
            # Charger les données cinématiques
            self.data = pd.read_csv(csv_file)
            print(f"✅ Données cinématiques chargées: {len(self.data)} points")
            
            # Charger la configuration depuis le fichier de résultats
            with open(config_file, 'r') as f:
                lines = f.readlines()
                
            # Parser les paramètres depuis le fichier config
            for line in lines:
                if "Rayon:" in line:
                    self.sphere_params['radius_mm'] = float(line.split(':')[1].strip().split()[0])
                elif "Masse:" in line:
                    self.sphere_params['mass_g'] = float(line.split(':')[1].strip().split()[0])
                elif "j =" in line:
                    self.sphere_params['j'] = float(line.split('j = ')[1].split(')')[0])
                elif "Densité:" in line:
                    self.sphere_params['density_kg_m3'] = float(line.split(':')[1].strip().split()[0])
                elif "FPS:" in line:
                    self.experimental_params['fps'] = float(line.split(':')[1].strip())
                elif "Angle:" in line and "°" in line:
                    angle_deg = float(line.split(':')[1].strip().split('°')[0])
                    self.experimental_params['angle_deg'] = angle_deg
                    self.experimental_params['angle_rad'] = np.radians(angle_deg)
            
            return True
        except Exception as e:
            print(f"❌ Erreur de chargement: {e}")
            return False
    
    def input_manual_parameters(self):
        """Saisie manuelle si les fichiers ne sont pas disponibles"""
        print("📝 SAISIE MANUELLE DES PARAMÈTRES")
        print("="*35)
        
        # Paramètres de la sphère
        self.sphere_params['radius_mm'] = float(input("Rayon de la sphère (mm): "))
        self.sphere_params['mass_g'] = float(input("Masse de la sphère (g): "))
        
        sphere_type = input("Type (1=solide, 2=creuse): ")
        self.sphere_params['j'] = 2/5 if sphere_type == "1" else 2/3
        
        # Paramètres expérimentaux
        self.experimental_params['fps'] = float(input("FPS de la caméra (250): ") or "250")
        self.experimental_params['angle_deg'] = float(input("Angle d'inclinaison (°): "))
        self.experimental_params['angle_rad'] = np.radians(self.experimental_params['angle_deg'])
        
        # Calculer la densité
        volume_mm3 = (4/3) * np.pi * self.sphere_params['radius_mm']**3
        volume_m3 = volume_mm3 * 1e-9
        mass_kg = self.sphere_params['mass_g'] * 1e-3
        self.sphere_params['density_kg_m3'] = mass_kg / volume_m3
    
    def calculate_advanced_kinematics(self):
        """Calculs cinématiques avancés"""
        print("\n🧮 CALCULS CINÉMATIQUES AVANCÉS")
        print("="*35)
        
        # Extraire les données
        t = self.data['time_s'].values
        x = self.data['x_m'].values
        y = self.data['y_m'].values
        vx = self.data['vx_ms'].values
        vy = self.data['vy_ms'].values
        v_mag = self.data['v_magnitude_ms'].values
        
        # Lisser les données pour réduire le bruit
        window_length = min(7, len(v_mag)//3)
        if window_length % 2 == 0:
            window_length += 1
        if window_length >= 3:
            v_smooth = savgol_filter(v_mag, window_length, 2)
        else:
            v_smooth = v_mag
        
        # 1. ACCÉLÉRATION
        dt = np.mean(np.diff(t))
        acceleration = np.gradient(v_smooth, dt)  # m/s²
        
        # 2. FORCES
        mass_kg = self.sphere_params['mass_g'] / 1000  # kg
        radius_m = self.sphere_params['radius_mm'] / 1000  # m
        
        # Force de résistance totale
        F_resistance = mass_kg * acceleration  # N (négatif = résistance)
        
        # Force gravitationnelle sur la pente
        angle = self.experimental_params['angle_rad']
        F_gravity = mass_kg * self.g * np.sin(angle)  # N
        
        # Force de résistance au roulement (corrigée de la gravité)
        F_rolling = F_resistance + F_gravity  # N
        
        # 3. ÉNERGIE
        # Énergie cinétique de translation
        E_trans = 0.5 * mass_kg * v_smooth**2  # J
        
        # Énergie cinétique de rotation
        I = self.sphere_params['j'] * mass_kg * radius_m**2  # kg.m²
        omega = v_smooth / radius_m  # rad/s (vitesse angulaire)
        E_rot = 0.5 * I * omega**2  # J
        
        # Énergie cinétique totale
        E_total = E_trans + E_rot  # J
        
        # Énergie potentielle (référence au début)
        h = (x - x[0]) * np.sin(angle)  # hauteur relative
        E_pot = mass_kg * self.g * h  # J
        
        # Énergie dissipée (par rapport au début)
        E_initial = E_total[0] + E_pot[0]
        E_dissipated = E_initial - (E_total + E_pot)  # J
        
        # 4. PUISSANCE
        # Puissance de résistance
        P_resistance = np.abs(F_rolling * v_smooth)  # W
        
        # Puissance gravitationnelle
        P_gravity = F_gravity * v_smooth  # W
        
        # 5. COEFFICIENTS
        # Coefficient de résistance instantané
        Krr_instantaneous = np.abs(F_rolling) / (mass_kg * self.g)
        
        # Stocker les résultats
        self.advanced_results = {
            'time': t,
            'position_x': x,
            'position_y': y,
            'velocity_smooth': v_smooth,
            'acceleration': acceleration,
            'F_resistance': F_resistance,
            'F_gravity': F_gravity,
            'F_rolling': F_rolling,
            'E_trans': E_trans,
            'E_rot': E_rot,
            'E_total': E_total,
            'E_pot': E_pot,
            'E_dissipated': E_dissipated,
            'P_resistance': P_resistance,
            'P_gravity': P_gravity,
            'Krr_instantaneous': Krr_instantaneous,
            'omega': omega
        }
        
        # Statistiques globales
        self.calculate_global_statistics()
        
        print(f"✅ Calculs terminés sur {len(t)} points")
    
    def calculate_global_statistics(self):
        """Calculer les statistiques globales"""
        # Moyennes et totaux
        results = self.advanced_results
        mass_kg = self.sphere_params['mass_g'] / 1000
        
        # Énergie totale dissipée
        total_energy_dissipated = results['E_dissipated'][-1]
        
        # Puissance moyenne
        avg_power_resistance = np.mean(results['P_resistance'])
        
        # Force de résistance moyenne
        avg_force_resistance = np.mean(np.abs(results['F_rolling']))
        
        # Coefficient Krr moyen
        avg_Krr = np.mean(results['Krr_instantaneous'])
        
        # Efficacité énergétique (énergie dissipée / énergie initiale)
        initial_energy = results['E_total'][0] + results['E_pot'][0]
        energy_efficiency = total_energy_dissipated / initial_energy * 100
        
        # Temps de décélération caractéristique
        v0 = results['velocity_smooth'][0]
        vf = results['velocity_smooth'][-1]
        avg_deceleration = np.mean(results['acceleration'])
        char_time = (vf - v0) / avg_deceleration if avg_deceleration != 0 else 0
        
        self.global_stats = {
            'total_energy_dissipated_J': total_energy_dissipated,
            'total_energy_dissipated_mJ': total_energy_dissipated * 1000,
            'avg_power_resistance_mW': avg_power_resistance * 1000,
            'avg_force_resistance_mN': avg_force_resistance * 1000,
            'avg_Krr': avg_Krr,
            'energy_efficiency_percent': energy_efficiency,
            'characteristic_time_s': char_time,
            'initial_energy_J': initial_energy,
            'final_energy_J': results['E_total'][-1] + results['E_pot'][-1]
        }
    
    def validate_theoretical_models(self):
        """Validation des modèles théoriques"""
        print("\n🔬 VALIDATION DES MODÈLES THÉORIQUES")
        print("="*40)
        
        # 1. Modèle Van wal (2017): Krr vs (1+j)⁻¹
        factor_1_plus_j_inv = 1 / (1 + self.sphere_params['j'])
        theoretical_krr_vanwal = 0.06 * factor_1_plus_j_inv  # Estimation basée sur la littérature
        
        # 2. Conservation de l'énergie
        initial_energy = self.global_stats['initial_energy_J']
        final_energy = self.global_stats['final_energy_J']
        energy_balance = (initial_energy - final_energy) / initial_energy * 100
        
        # 3. Relation puissance-vitesse
        v = self.advanced_results['velocity_smooth']
        P = self.advanced_results['P_resistance']
        
        # Ajustement P = a * v^b
        valid_indices = (v > 0.001) & (P > 0)
        if np.sum(valid_indices) > 3:
            log_v = np.log(v[valid_indices])
            log_P = np.log(P[valid_indices])
            slope, intercept, r_value, _, _ = stats.linregress(log_v, log_P)
            power_law_exponent = slope
            power_law_coefficient = np.exp(intercept)
            power_law_r2 = r_value**2
        else:
            power_law_exponent = 0
            power_law_coefficient = 0
            power_law_r2 = 0
        
        self.theoretical_validation = {
            'factor_1_plus_j_inv': factor_1_plus_j_inv,
            'theoretical_krr_vanwal': theoretical_krr_vanwal,
            'measured_krr': self.global_stats['avg_Krr'],
            'krr_deviation_percent': abs(theoretical_krr_vanwal - self.global_stats['avg_Krr']) / theoretical_krr_vanwal * 100,
            'energy_balance_percent': energy_balance,
            'power_law_exponent': power_law_exponent,
            'power_law_coefficient': power_law_coefficient,
            'power_law_r2': power_law_r2
        }
        
        print(f"✅ Facteur (1+j)⁻¹: {factor_1_plus_j_inv:.4f}")
        print(f"✅ Krr théorique (Van wal): {theoretical_krr_vanwal:.4f}")
        print(f"✅ Krr mesuré: {self.global_stats['avg_Krr']:.4f}")
        print(f"✅ Écart: {self.theoretical_validation['krr_deviation_percent']:.1f}%")
        print(f"✅ Bilan énergétique: {energy_balance:.1f}%")
    
    def plot_advanced_analysis(self):
        """Créer les graphiques d'analyse avancée"""
        print("\n📊 GÉNÉRATION DES GRAPHIQUES AVANCÉS")
        print("="*40)
        
        # Vérifications de sécurité et debug
        if not self.advanced_results:
            print("❌ Pas de résultats avancés à tracer")
            return
            
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # Extraire et vérifier les données
        try:
            t = self.advanced_results['time']
            v_smooth = self.advanced_results['velocity_smooth']
            acceleration = self.advanced_results['acceleration']
            
            print(f"🔍 Debug - Longueurs des données:")
            print(f"   t: {len(t)}")
            print(f"   v_smooth: {len(v_smooth)}")
            print(f"   acceleration: {len(acceleration)}")
            
            # Trouver la longueur minimale de façon sécurisée
            lengths = [len(t), len(v_smooth), len(acceleration)]
            min_length = min(lengths)
            print(f"   min_length: {min_length}")
            
            if min_length < 3:
                print("❌ Pas assez de données pour tracer")
                return
                
            # Tronquer TOUTES les données à la même longueur
            t_plot = np.array(t[:min_length])
            v_plot = np.array(v_smooth[:min_length])
            a_plot = np.array(acceleration[:min_length])
            
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction des données: {e}")
            return
        
        axes[0, 0].plot(t_plot, v_plot*1000, 'b-', linewidth=2, label='Vitesse (mm/s)')
        ax_acc = axes[0, 0].twinx()
        ax_acc.plot(t_plot, a_plot*1000, 'r-', linewidth=2, label='Accélération (mm/s²)')
        axes[0, 0].set_xlabel('Temps (s)')
        axes[0, 0].set_ylabel('Vitesse (mm/s)', color='b')
        ax_acc.set_ylabel('Accélération (mm/s²)', color='r')
        axes[0, 0].set_title('Vitesse et Accélération')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Forces
        F_grav = self.advanced_results['F_gravity'][:min_length]
        F_roll = self.advanced_results['F_rolling'][:min_length]
        axes[0, 1].plot(t_plot, F_grav*1000, 'g-', linewidth=2, label='F_gravity')
        axes[0, 1].plot(t_plot, np.abs(F_roll)*1000, 'r-', linewidth=2, label='|F_rolling|')
        axes[0, 1].set_xlabel('Temps (s)')
        axes[0, 1].set_ylabel('Force (mN)')
        axes[0, 1].set_title('Forces')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Énergies
        E_trans = self.advanced_results['E_trans'][:min_length]
        E_rot = self.advanced_results['E_rot'][:min_length]
        E_total = self.advanced_results['E_total'][:min_length]
        axes[0, 2].plot(t_plot, E_trans*1000, 'b-', linewidth=2, label='E_translation')
        axes[0, 2].plot(t_plot, E_rot*1000, 'r-', linewidth=2, label='E_rotation')
        axes[0, 2].plot(t_plot, E_total*1000, 'k-', linewidth=2, label='E_totale')
        axes[0, 2].set_xlabel('Temps (s)')
        axes[0, 2].set_ylabel('Énergie (mJ)')
        axes[0, 2].set_title('Énergies Cinétiques')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Bilan énergétique
        E_pot = self.advanced_results['E_pot'][:min_length]
        E_dissipated = self.advanced_results['E_dissipated'][:min_length]
        E_mech = E_total + E_pot
        axes[1, 0].plot(t_plot, E_mech*1000, 'b-', linewidth=2, label='E_mécanique')
        axes[1, 0].plot(t_plot, E_dissipated*1000, 'r-', linewidth=2, label='E_dissipée')
        axes[1, 0].set_xlabel('Temps (s)')
        axes[1, 0].set_ylabel('Énergie (mJ)')
        axes[1, 0].set_title('Bilan Énergétique')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Puissance
        P_res = self.advanced_results['P_resistance'][:min_length]
        P_grav = self.advanced_results['P_gravity'][:min_length]
        axes[1, 1].plot(t_plot, P_res*1000, 'r-', linewidth=2, label='P_résistance')
        axes[1, 1].plot(t_plot, P_grav*1000, 'g-', linewidth=2, label='P_gravité')
        axes[1, 1].set_xlabel('Temps (s)')
        axes[1, 1].set_ylabel('Puissance (mW)')
        axes[1, 1].set_title('Puissances')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Krr instantané
        Krr_inst = self.advanced_results['Krr_instantaneous'][:min_length]
        axes[1, 2].plot(t_plot, Krr_inst, 'purple', linewidth=2)
        axes[1, 2].axhline(y=self.global_stats['avg_Krr'], color='orange', linestyle='--', 
                          label=f'Moyenne: {self.global_stats["avg_Krr"]:.4f}')
        axes[1, 2].set_xlabel('Temps (s)')
        axes[1, 2].set_ylabel('Krr instantané')
        axes[1, 2].set_title('Coefficient Krr Instantané')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Relation Puissance-Vitesse
        v = v_plot  # Utiliser les données tronquées
        P = P_res   # Utiliser les données tronquées
        valid = (v > 0.001) & (P > 0)
        if np.sum(valid) > 3:
            axes[2, 0].scatter(v[valid]*1000, P[valid]*1000, alpha=0.6, s=20)
            axes[2, 0].set_xlabel('Vitesse (mm/s)')
            axes[2, 0].set_ylabel('Puissance (mW)')
            axes[2, 0].set_title(f'P vs V (R² = {self.theoretical_validation["power_law_r2"]:.3f})')
            axes[2, 0].set_xscale('log')
            axes[2, 0].set_yscale('log')
            axes[2, 0].grid(True, alpha=0.3)
        else:
            axes[2, 0].text(0.5, 0.5, 'Données insuffisantes\npour P vs V', 
                           ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 0].set_title('P vs V (données insuffisantes)')
            axes[2, 0].set_xlabel('Vitesse (mm/s)')
            axes[2, 0].set_ylabel('Puissance (mW)')
        
        # 8. Validation Van wal
        measured_krr = self.global_stats['avg_Krr']
        theoretical_krr = self.theoretical_validation['theoretical_krr_vanwal']
        axes[2, 1].bar(['Théorique\n(Van wal)', 'Mesuré'], [theoretical_krr, measured_krr], 
                      color=['lightblue', 'orange'], alpha=0.7)
        axes[2, 1].set_ylabel('Coefficient Krr')
        axes[2, 1].set_title('Validation Van wal (2017)')
        axes[2, 1].grid(True, alpha=0.3, axis='y')
        
        # 9. Résumé statistiques
        stats_text = f"""STATISTIQUES GLOBALES:

Énergie dissipée: {self.global_stats['total_energy_dissipated_mJ']:.2f} mJ
Puissance moyenne: {self.global_stats['avg_power_resistance_mW']:.2f} mW
Force résistance: {self.global_stats['avg_force_resistance_mN']:.2f} mN
Krr moyen: {self.global_stats['avg_Krr']:.6f}
Efficacité: {self.global_stats['energy_efficiency_percent']:.1f}%

VALIDATION:
Facteur (1+j)⁻¹: {self.theoretical_validation['factor_1_plus_j_inv']:.4f}
Écart Van wal: {self.theoretical_validation['krr_deviation_percent']:.1f}%
Loi puissance: P ∝ V^{self.theoretical_validation['power_law_exponent']:.2f}"""
        
        axes[2, 2].text(0.05, 0.95, stats_text, transform=axes[2, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[2, 2].set_title('Résumé des Résultats')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig("results/advanced_kinematics_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Graphiques sauvegardés dans results/advanced_kinematics_analysis.png")
    
    def save_advanced_results(self):
        """Sauvegarder les résultats avancés"""
        # Sauvegarder les données détaillées
        detailed_data = pd.DataFrame(self.advanced_results)
        detailed_data.to_csv("results/advanced_kinematics_data.csv", index=False)
        
        # Rapport détaillé
        with open("results/advanced_kinematics_report.txt", "w") as f:
            f.write("ANALYSE CINÉMATIQUE AVANCÉE\n")
            f.write("="*50 + "\n\n")
            
            f.write("PARAMÈTRES PHYSIQUES:\n")
            f.write("-"*20 + "\n")
            f.write(f"Masse: {self.sphere_params['mass_g']} g\n")
            f.write(f"Rayon: {self.sphere_params['radius_mm']} mm\n")
            f.write(f"Facteur j: {self.sphere_params['j']}\n")
            f.write(f"Angle: {self.experimental_params['angle_deg']}°\n\n")
            
            f.write("RÉSULTATS ÉNERGÉTIQUES:\n")
            f.write("-"*22 + "\n")
            f.write(f"Énergie initiale: {self.global_stats['initial_energy_J']*1000:.3f} mJ\n")
            f.write(f"Énergie finale: {self.global_stats['final_energy_J']*1000:.3f} mJ\n")
            f.write(f"Énergie dissipée: {self.global_stats['total_energy_dissipated_mJ']:.3f} mJ\n")
            f.write(f"Efficacité énergétique: {self.global_stats['energy_efficiency_percent']:.1f}%\n\n")
            
            f.write("RÉSULTATS DYNAMIQUES:\n")
            f.write("-"*20 + "\n")
            f.write(f"Force résistance moyenne: {self.global_stats['avg_force_resistance_mN']:.3f} mN\n")
            f.write(f"Puissance résistance moyenne: {self.global_stats['avg_power_resistance_mW']:.3f} mW\n")
            f.write(f"Coefficient Krr moyen: {self.global_stats['avg_Krr']:.6f}\n")
            f.write(f"Temps caractéristique: {self.global_stats['characteristic_time_s']:.3f} s\n\n")
            
            f.write("VALIDATION THÉORIQUE:\n")
            f.write("-"*19 + "\n")
            f.write(f"Facteur (1+j)⁻¹: {self.theoretical_validation['factor_1_plus_j_inv']:.4f}\n")
            f.write(f"Krr théorique (Van wal): {self.theoretical_validation['theoretical_krr_vanwal']:.6f}\n")
            f.write(f"Écart avec théorie: {self.theoretical_validation['krr_deviation_percent']:.1f}%\n")
            f.write(f"Bilan énergétique: {self.theoretical_validation['energy_balance_percent']:.1f}%\n")
            f.write(f"Loi puissance P ∝ V^{self.theoretical_validation['power_law_exponent']:.2f} (R² = {self.theoretical_validation['power_law_r2']:.3f})\n")
        
        print("✅ Résultats sauvegardés:")
        print("   - results/advanced_kinematics_data.csv")
        print("   - results/advanced_kinematics_report.txt")
    
    def run_complete_advanced_analysis(self):
        """Lancer l'analyse complète"""
        print("🚀 ANALYSE CINÉMATIQUE AVANCÉE")
        print("="*50)
        
        # Étape 1: Chargement des données
        print("\n📊 ÉTAPE 1: Chargement des données")
        if not self.load_existing_analysis():
            print("⚠️ Chargement automatique échoué. Saisie manuelle:")
            self.input_manual_parameters()
            
            # Charger les données de base
            try:
                self.data = pd.read_csv("results/kinematics_analysis.csv")
                print(f"✅ Données de base chargées: {len(self.data)} points")
            except:
                print("❌ Impossible de charger les données cinématiques")
                return False
        
        # Étape 2: Calculs avancés
        print("\n🧮 ÉTAPE 2: Calculs cinématiques avancés")
        self.calculate_advanced_kinematics()
        
        # Étape 3: Validation théorique
        print("\n🔬 ÉTAPE 3: Validation des modèles")
        self.validate_theoretical_models()
        
        # Étape 4: Visualisation
        print("\n📊 ÉTAPE 4: Génération des graphiques")
        self.plot_advanced_analysis()
        
        # Étape 5: Sauvegarde
        print("\n💾 ÉTAPE 5: Sauvegarde des résultats")
        self.save_advanced_results()
        
        print("\n🎉 ANALYSE AVANCÉE TERMINÉE!")
        print(f"📋 Résumé: Énergie dissipée = {self.global_stats['total_energy_dissipated_mJ']:.2f} mJ")
        print(f"🔬 Validation: Krr = {self.global_stats['avg_Krr']:.6f} (écart = {self.theoretical_validation['krr_deviation_percent']:.1f}%)")
        
        return True

# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main():
    """Fonction principale pour lancer l'analyse avancée"""
    analyzer = AdvancedKinematicsAnalyzer()
    
    # Vérifier les fichiers nécessaires
    import os
    required_files = ["results/kinematics_analysis.csv", "results/rolling_resistance_analysis.txt"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("⚠️ Fichiers manquants:")
        for f in missing_files:
            print(f"   - {f}")
        print("   Lancez d'abord l'analyse de base ou saisissez les paramètres manuellement.")
    
    # Lancer l'analyse
    analyzer.run_complete_advanced_analysis()

# Lancer le script
if __name__ == "__main__":
    main()
