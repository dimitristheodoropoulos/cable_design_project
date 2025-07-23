#!/usr/bin/env python
# coding: utf-8
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import electricpy as ep
import json
import os
import skrf as rf
from skrf.media import coaxial

print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {plt.matplotlib.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"SciPy version: {scipy.__version__}")
print(f"ElectricPy version: {ep.__version__}")

# Simple plot to confirm matplotlib is functional
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(8, 4))
plt.plot(x, y)
plt.title("Matplotlib Test Plot")
plt.xlabel("X")
plt.ylabel("sin(X)")
plt.grid(True)
plt.savefig("matplotlib_test_plot.png")
plt.show()

print("\\nEnvironment setup confirmed!")


# --- Material Properties Loading ---

MATERIALS_FILE_PATH = "materials.json"

def load_material_properties(file_path):
    
    try:
        with open(file_path, 'r') as f:
            materials = json.load(f)
        print(f"Material properties loaded successfully from {file_path}")
        return materials
    except FileNotFoundError:
        print(f"Error: Material properties file not found at {file_path}. Please create it.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return {}

# --- CableDesigner Class ---
class CableDesigner:
    EPSILON_0 = 8.854e-12 # Permittivity of free space (F/m)

    def __init__(self, materials_data, conductor_material, conductor_diameter_mm,
                 insulation_material, insulation_thickness_mm, sheath_material, sheath_thickness_mm,
                 armor_material=None, armor_thickness_mm=0.0,
                 shield_material=None, shield_thickness_mm=0.0,
                 shield_type=None, shield_coverage_percentage=100.0,
                 operating_voltage_kV=11.0,
                 operating_frequency_hz=50.0,
                 num_conductors=3,
                 # ΠΡΟΣΘΕΣΤΕ ΕΔΩ ΤΙΣ ΠΑΡΑΜΕΤΡΟΥΣ ΤΕΡΜΑΤΙΣΜΟΥ
                 num_terminations=2, # Αριθμός τερματισμών (π.χ., 2 για μια καλωδιακή διαδρομή)
                 termination_loss_per_termination_W=10.0, # Απώλειες σε Watts ανά μονάδα τερματισμού
                 laying_method='direct_buried', # 'direct_buried', 'in_ducts', 'in_air'
                 duct_bank_config=None, # e.g., {'rows': 1, 'cols': 1, 'spacing_m': 0.3}
                 thermal_resistivity_soil_Km_W=1.0, # for direct_buried or in_ducts
                 ambient_air_temp_C=30.0, # for in_air
                 solar_radiation_W_m2=1000.0, # for in_air
                 wind_speed_m_s=0.5, # for in_air
                 ground_temp_C=20.0, # for direct_buried
                 burial_depth_m=0.8, # <--- ADD THIS PARAMETER HERE with a default value
                 shield_bonding='single_point', # 'single_point', 'multi_point', 'cross_bonded'
                 sccr_kA=50, # Short Circuit Current Rating in kA
                 installation_temp_C=25.0, # for bending radius calculation
                 min_bending_radius_multiplier=12, # typically 12-20 for power cables
                 load_factor=1.0, # ratio of actual load to rated load
                 losses_pct=5.0, # total losses in percentage
                 length_km=1.0 # Προσθέστε αυτή την παράμετρο με μια προεπιλεγμένη τιμή
                ):

        self.burial_depth_m = burial_depth_m
        self.materials = materials_data
        self.length_km = length_km
        self.length = length_km * 1000  # Μήκος σε μέτρα
        self.conductor_material = conductor_material
        self.conductor_diameter_mm = conductor_diameter_mm
        self.insulation_material = insulation_material
        self.insulation_thickness_mm = insulation_thickness_mm
        self.sheath_material = sheath_material
        self.sheath_thickness_mm = sheath_thickness_mm
        self.armor_material = armor_material
        self.armor_thickness_mm = armor_thickness_mm
        self.shield_material = shield_material
        self.shield_thickness_mm = shield_thickness_mm
        self.shield_type = shield_type
        self.shield_coverage_percentage = shield_coverage_percentage / 100.0 # Convert to ratio
        self.operating_voltage_kV = operating_voltage_kV
        self.operating_frequency_hz = operating_frequency_hz
        self.num_conductors = num_conductors
        self.num_terminations = num_terminations
        self.termination_loss_per_termination_W = termination_loss_per_termination_W
        self.laying_method = laying_method
        self.duct_bank_config = duct_bank_config
        self.thermal_resistivity_soil_Km_W = thermal_resistivity_soil_Km_W
        self.ambient_air_temp_C = ambient_air_temp_C
        self.solar_radiation_W_m2 = solar_radiation_W_m2
        self.wind_speed_m_s = wind_speed_m_s
        self.ground_temp_C = ground_temp_C
        self.shield_bonding = shield_bonding
        self.sccr_kA = sccr_kA
        self.installation_temp_C = installation_temp_C
        self.min_bending_radius_multiplier = min_bending_radius_multiplier
        self.load_factor = load_factor
        self.losses_pct = losses_pct

        self._validate_materials()
        self._calculate_geometric_properties()

    def _validate_materials(self):
        required_materials = [self.conductor_material, self.insulation_material, self.sheath_material]
        if self.armor_material:
            required_materials.append(self.armor_material)
        if self.shield_material:
            required_materials.append(self.shield_material)

        for material in required_materials:
            if material not in self.materials:
                raise ValueError(f"Material '{material}' not found in material properties data.")

    def _calculate_geometric_properties(self):
        # Conductor radius
        self.r_conductor_m = (self.conductor_diameter_mm / 2) / 1000.0

        # Insulation outer radius
        self.r_insulation_outer_m = self.r_conductor_m + (self.insulation_thickness_mm / 1000.0)

        # Sheath outer radius
        self.r_sheath_outer_m = self.r_insulation_outer_m + (self.sheath_thickness_mm / 1000.0)

        # Shield outer radius (if shield exists)
        self.r_shield_outer_m = 0.0
        if self.shield_material and self.shield_thickness_mm > 0:
            self.r_shield_outer_m = self.r_sheath_outer_m + (self.shield_thickness_mm / 1000.0)
        else:
            self.r_shield_outer_m = self.r_sheath_outer_m # If no shield, shield outer is sheath outer

        # Armor outer radius (if armor exists)
        self.r_armor_outer_m = 0.0
        if self.armor_material and self.armor_thickness_mm > 0:
            # If shield exists, armor is applied over the shield
            if self.r_shield_outer_m > self.r_sheath_outer_m:
                self.r_armor_outer_m = self.r_shield_outer_m + (self.armor_thickness_mm / 1000.0)
            else: # Armor is applied over the sheath
                self.r_armor_outer_m = self.r_sheath_outer_m + (self.armor_thickness_mm / 1000.0)
        else:
            self.r_armor_outer_m = max(self.r_sheath_outer_m, self.r_shield_outer_m) # If no armor, use largest existing outer radius

        self.overall_diameter_m = self.r_armor_outer_m * 2


    def calculate_resistance(self, temperature_C):
        rho_ref = self.materials[self.conductor_material]['resistivity_at_20C_ohm_m']
        alpha = self.materials[self.conductor_material]['alpha_temp_coeff']
        area_conductor_m2 = np.pi * (self.r_conductor_m**2)
        
        # Adjust resistivity for temperature
        rho_temp = rho_ref * (1 + alpha * (temperature_C - 20))
        
        # Resistance per unit length (Ohm/m)
        resistance_per_meter = rho_temp / area_conductor_m2
        return resistance_per_meter

    def calculate_inductance(self):
        # Internal inductance (simplified, often negligible for power frequencies)
        L_internal = self.materials[self.conductor_material]['permeability_relative'] * 2e-7 / 2 # H/m

        # External inductance for single core cable (approximated)
        # Assuming concentric layers. For more rigorous, use partial inductances.
        # Here, Dm is geometric mean radius, GMR for single conductor is r' = r * e^(-1/4)
        r_prime = self.r_conductor_m * np.exp(-1/4)
        
        # Consider the return path at a large distance for calculation purposes
        # For a single-core cable, inductance calculation often considers a distant return path or GMR/GMD
        # A more practical approach for single core in isolation:
        # L = (mu_0 / 2*pi) * ln(D/r'), where D is distance to return path (large)
        # Since this is a single CableDesigner, we'll calculate self-inductance and assume typical GMR methods.
        
        # For a single core cable, the external inductance due to flux between conductor and sheath/shield.
        # This is a common formula for coaxial or single conductor within a metallic return path.
        if self.r_shield_outer_m > self.r_conductor_m:
            L_external = (2e-7) * np.log(self.r_shield_outer_m / self.r_conductor_m) # H/m (for non-magnetic medium)
        else: # No effective concentric shield or return
            L_external = (2e-7) * np.log(1000 / self.r_conductor_m) # Arbitrary large distance for isolated conductor approx (H/m)
        
        return L_internal + L_external

    def calculate_capacitance(self):
        # Capacitance per unit length (F/m)
        # For a single core cable with concentric layers
        r_inner = self.r_conductor_m
        r_outer = self.r_insulation_outer_m
        
        epsilon_r = self.materials[self.insulation_material]['dielectric_constant_relative']
        capacitance_per_meter = (2 * np.pi * CableDesigner.EPSILON_0 * epsilon_r) / np.log(r_outer / r_inner)
        return capacitance_per_meter

    def calculate_conductance(self):
        # Conductance per unit length (S/m) due to dielectric losses
        C = self.calculate_capacitance()
        tan_delta = self.materials[self.insulation_material]['dissipation_factor_tan_delta']
        G = 2 * np.pi * self.operating_frequency_hz * C * tan_delta
        return G
    
    def calculate_shield_losses(self, current_A, conductor_temp_C):
        """
        Calculates losses in the shield (circulating and eddy currents) per meter.
        These are typically expressed as a factor of the conductor losses.
        Reference: IEC 60287-1-1, Section 2.2.3 (Losses in screens)
        
        Assumes losses are I^2 * R * (1 + lambda_0 + lambda_1 + lambda_2),
        where lambda_0 are circulating current losses (bonding dependent),
        and lambda_1, lambda_2 are eddy current losses in shield/armor respectively.
        
        For simplicity, we use a k_shield_circulating_loss_factor from materials.json
        which conceptually includes all shield-related losses as a fraction of conductor losses.
        """
        if not self.shield_material or self.shield_thickness_mm == 0:
            return 0.0 # No shield, no shield losses

        # Get relevant properties for shield material
        shield_props = self.materials[self.shield_material]
        
        # Check if k_shield_circulating_loss_factor is a dictionary (for bonding types)
        if isinstance(shield_props.get('k_shield_circulating_loss_factor'), dict):
            k_shield_loss_factor = shield_props['k_shield_circulating_loss_factor'].get(self.shield_bonding)
            if k_shield_loss_factor is None:
                print(f"Warning: k_shield_circulating_loss_factor for '{self.shield_bonding}' bonding not found for {self.shield_material}. Falling back to generic factor if available.")
                k_shield_loss_factor = shield_props['k_shield_circulating_loss_factor'].get('default') # Look for a 'default' key if it's a dict
        else: # It's a direct float value
            k_shield_loss_factor = shield_props.get('k_shield_circulating_loss_factor')
        
        if k_shield_loss_factor == None:
            print(f"Warning: k_shield_circulating_loss_factor not found for {self.shield_material} (or for bonding type {self.shield_bonding}). Assuming 0 shield losses.")
            return 0.0

        # Conductor AC resistance at operating temperature
        R_ac_conductor_ohm_m = self.calculate_resistance(conductor_temp_C)

        # Conductor losses (3-phase) per meter
        # Assuming current_A is line current, and num_conductors is number of phases
        conductor_losses_W_m = (current_A**2) * R_ac_conductor_ohm_m * self.num_conductors

        # Shield losses are a fraction of conductor losses
        shield_losses_W_m = conductor_losses_W_m * k_shield_loss_factor
        
        # Consider coverage for partial shields (e.g., tape with gaps)
        shield_losses_W_m *= (self.shield_coverage_percentage)

        return shield_losses_W_m

    def calculate_thermal_resistance(self, sheath_temp_C, ambient_temp_C, burial_depth_m=None,
                                     apply_termination_effect=False, termination_length_m=1.0):
        # Simplified thermal resistance calculation
        # This is highly dependent on laying method and detailed geometry.
        # Here, we provide a basic model. For precise results, FEA or detailed standards are needed.

        R_ins = (1 / (2 * np.pi * self.materials[self.insulation_material]['thermal_conductivity_W_mK'])) * \
                np.log(self.r_insulation_outer_m / self.r_conductor_m) # K*m/W per conductor

        R_sheath = (1 / (2 * np.pi * self.materials[self.sheath_material]['thermal_conductivity_W_mK'])) * \
                   np.log(self.r_sheath_outer_m / self.r_insulation_outer_m) # K*m/W per cable

        R_armor = 0.0
        if self.armor_material and self.armor_thickness_mm > 0:
            # Thermal resistance of armor layer
            R_armor = (1 / (2 * np.pi * self.materials[self.armor_material]['thermal_conductivity_W_mK'])) * \
                      np.log(self.r_armor_outer_m / max(self.r_sheath_outer_m, self.r_shield_outer_m))

        R_external = 0.0
        if self.laying_method == 'direct_buried' and burial_depth_m is not None:
            # IEC 60287 approximation for direct buried
            rho_soil = self.thermal_resistivity_soil_Km_W
            D_e = self.overall_diameter_m # External diameter of the cable
            L = burial_depth_m # Depth to center of cable
            R_external = (rho_soil / (2 * np.pi)) * np.arccosh(2 * L / D_e)
        elif self.laying_method == 'in_air':
            # Simplified convection/radiation for in air (very complex in reality)
            # This is a placeholder; detailed heat transfer needed for accuracy
            h_conv = 10 # W/m2.K, typical for natural convection in air
            epsilon_surface = 0.9 # Emissivity of outer jacket
            sigma_sb = 5.67e-8 # Stefan-Boltzmann constant
            
            # Assuming cable surface temp is close to sheath temp for this simplified model
            T_surface = sheath_temp_C + 273.15 # K
            T_ambient_air = ambient_temp_C + 273.15 # K

            # Q_rad = epsilon_surface * sigma_sb * np.pi * self.overall_diameter_m * (T_surface**4 - T_ambient_air**4) # W/m
            # Q_conv = h_conv * np.pi * self.overall_diameter_m * (T_surface - T_ambient_air) # W/m

            # Total heat dissipation capacity per meter per degree C temp difference (approx)
            # This isn't a simple thermal resistance. Let's provide a "thermal impedance" instead.
            # A simpler approach: assume a fixed external thermal resistance based on general values
            R_external = 1.0 # K*m/W, a rough estimate for cables in air

        # Total thermal resistance from conductor to ambient
        R_total = R_ins + R_sheath + R_armor + R_external # K*m/W (or C*m/W)

        return R_total # ΕΠΙΣΤΡΕΦΕΤΕ ΜΟΝΟ ΤΟ R_total

    def calculate_current_capacity(self, conductor_temp_C, ambient_temp_C, length_km,
                                   burial_depth_m=None, apply_termination_effect=False,
                                   ground_temp_C=None, ambient_air_temp_C_in_air=None):
        
        # Calculate maximum permissible temperature rise
        delta_theta = conductor_temp_C - ambient_temp_C # Simplified delta_theta (can be more complex)

        # Get total thermal resistance
        R_thermal_total = self.calculate_thermal_resistance(
            conductor_temp_C,  # Περάστε το ως positional argument
            ambient_temp_C,    # Περάστε το ως positional argument
            burial_depth_m=burial_depth_m,
            apply_termination_effect=apply_termination_effect
        )

        # Conductor AC Resistance at operating temperature
        R_ac_ohm_m = self.calculate_resistance(conductor_temp_C)
        
        # If the thermal resistance is effectively zero or negative, it indicates an issue or a non-physical scenario.
        if R_thermal_total <= 0:
            print("Warning: Calculated total thermal resistance is non-positive, cannot determine ampacity.")
            calculated_termination_heat_W_per_km = 0.0 # Initialize to 0.0 for safety
            # Return nested structure even for error case
            return {
                'ampacity_results': {
                    'ampacity_A': 0.0,
                    'conductor_temp_at_ampacity': conductor_temp_C,
                    'total_heat_dissipated_W_per_m': 0.0,
                    'thermal_resistance_total_Km_W': R_thermal_total,
                    'resistance_ac_ohm_m': R_ac_ohm_m,
                    'dielectric_losses_W_per_m': 0.0,
                    'shield_losses_W_per_m': 0.0,
                    'termination_heat_W_per_km': calculated_termination_heat_W_per_km,
                },
                'compliance_checks': self._run_compliance_checks(0.0, {'max_tensile_force_N': 0, 'max_compressive_force_N': 0}) # Pass dummy mechanical data
            }

        # Dielectric losses (W/m)
        V_phase = self.operating_voltage_kV * 1000 / np.sqrt(3) # Volts
        W_d = self.calculate_conductance() * V_phase**2 * self.num_conductors # Total dielectric loss per meter for all conductors

        # Total allowable temperature rise (theta_c - theta_a)
        Delta_Theta = conductor_temp_C - ambient_temp_C

        # Adjusted denominator to include shield losses as an effective increase in conductor resistance losses.
        # This is a standard method (e.g., IEC 60287) to account for shield losses as a fraction of conductor losses
        # in the thermal circuit for ampacity calculation.
        effective_k_shield_loss_factor = 0.0
        if self.shield_material and self.shield_thickness_mm > 0:
            shield_props = self.materials[self.shield_material]
            if isinstance(shield_props.get('k_shield_circulating_loss_factor'), dict):
                effective_k_shield_loss_factor = shield_props['k_shield_circulating_loss_factor'].get(self.shield_bonding, 0.0)
                if effective_k_shield_loss_factor == 0.0: # Fallback to generic if specific bonding not found
                    effective_k_shield_loss_factor = shield_props['k_shield_circulating_loss_factor'].get('default', 0.0)
            else:
                effective_k_shield_loss_factor = shield_props.get('k_shield_circulating_loss_factor', 0.0)
            effective_k_shield_loss_factor *= (self.shield_coverage_percentage) # Apply coverage

        # Υπολογισμός θερμότητας τερματισμού δυναμικά για αναφορά, εάν εφαρμόζεται
        calculated_termination_heat_W_per_km = 0.0
        if apply_termination_effect and length_km > 0:
            total_termination_losses_W = self.num_terminations * self.num_conductors * self.termination_loss_per_termination_W
            calculated_termination_heat_W_per_km = total_termination_losses_W / (length_km) # W/km

        # Original calculation for numerator
        # We ensure termination heat is NOT subtracted here, as it's a localized effect
        # not suitable for a uniform distributed heat loss model for ampacity.
        numerator = Delta_Theta - (W_d * R_thermal_total)
        denominator = R_ac_ohm_m * R_thermal_total * self.num_conductors * (1 + effective_k_shield_loss_factor)

        # Ensure numerator is positive for real current
        if numerator < 0:
            print(f"Warning: Dielectric losses ({W_d:.2f} W/m) are too high for the given temperature difference ({Delta_Theta:.2f} C) and thermal resistance ({R_thermal_total:.2f} Km/W). Ampacity cannot be calculated accurately with these parameters. Setting ampacity to 0.")
            # Return nested structure even for error case
            return {
                'ampacity_results': {
                    'ampacity_A': 0.0,
                    'conductor_temp_at_ampacity': conductor_temp_C,
                    'total_heat_dissipated_W_per_m': 0.0,
                    'thermal_resistance_total_Km_W': R_thermal_total,
                    'resistance_ac_ohm_m': R_ac_ohm_m,
                    'dielectric_losses_W_per_m': W_d,
                    'shield_losses_W_per_m': 0.0,
                    'termination_heat_W_per_km': calculated_termination_heat_W_per_km, # Still report if calculated
                },
                'compliance_checks': self._run_compliance_checks(0.0, {'max_tensile_force_N': 0, 'max_compressive_force_N': 0}) # Pass dummy mechanical data
            }


        # Ampacity calculation
        ampacity_A = np.sqrt(numerator / denominator)
        
        # Calculate total heat dissipated for ampacity
        total_heat_dissipated_W_per_m = (ampacity_A**2 * R_ac_ohm_m * self.num_conductors * (1 + effective_k_shield_loss_factor)) + W_d

        shield_losses_at_ampacity_W_per_m = (ampacity_A**2) * R_ac_ohm_m * self.num_conductors * effective_k_shield_loss_factor

        # Dummy mechanical results for initial run
        mechanical_results_dummy = self._calculate_mechanical_properties()

        # Return the results in a nested dictionary
        ampacity_metrics = {
            'ampacity_A': ampacity_A,
            'conductor_temp_at_ampacity': conductor_temp_C,
            'total_heat_dissipated_W_per_m': total_heat_dissipated_W_per_m,
            'thermal_resistance_total_Km_W': R_thermal_total,
            'resistance_ac_ohm_m': R_ac_ohm_m,
            'dielectric_losses_W_per_m': W_d,
            'shield_losses_W_per_m': shield_losses_at_ampacity_W_per_m,
            'termination_heat_W_per_km': calculated_termination_heat_W_per_km,
        }
        
        return {
            'ampacity_results': ampacity_metrics,
            'compliance_checks': self._run_compliance_checks(ampacity_A, mechanical_results_dummy)
        }

    def calculate_voltage_drop(self, current_A, power_factor=0.9):
        # Calculate impedance per unit length
        R = self.calculate_resistance(self.materials[self.conductor_material]['max_operating_temp_C'])
        X_L = 2 * np.pi * self.operating_frequency_hz * self.calculate_inductance()
        
        Z = np.sqrt(R**2 + X_L**2) # Ohms/meter
        
        # Voltage drop per unit length (Volt/meter)
        # For 3-phase system, line-to-line voltage drop
        # Simplified formula: V_drop_LL = sqrt(3) * I * (R*cos(phi) + X*sin(phi)) * L
        
        voltage_drop_per_meter = self.num_conductors * current_A * (R * power_factor + X_L * np.sin(np.arccos(power_factor)))
        total_voltage_drop_V = voltage_drop_per_meter * self.length # Total voltage drop in Volts

        return total_voltage_drop_V

    def calculate_cost(self):
        cost_conductor = (np.pi * self.r_conductor_m**2 * self.length *
                          self.materials[self.conductor_material]['density_kg_m3'] *
                          self.materials[self.conductor_material]['cost_per_kg_usd']) * self.num_conductors

        cost_insulation = (np.pi * (self.r_insulation_outer_m**2 - self.r_conductor_m**2) * self.length *
                           self.materials[self.insulation_material]['density_kg_m3'] *
                           self.materials[self.insulation_material]['cost_per_kg_usd']) * self.num_conductors
        
        cost_sheath = (np.pi * (self.r_sheath_outer_m**2 - self.r_insulation_outer_m**2) * self.length *
                       self.materials[self.sheath_material]['density_kg_m3'] *
                       self.materials[self.sheath_material]['cost_per_kg_usd'])

        cost_shield = 0.0
        if self.shield_material and self.shield_thickness_mm > 0:
            # Assuming shield is over sheath
            cost_shield = (np.pi * (self.r_shield_outer_m**2 - self.r_sheath_outer_m**2) * self.length *
                           self.materials[self.shield_material]['density_kg_m3'] *
                           self.materials[self.shield_material]['cost_per_kg_usd'])

        cost_armor = 0.0 # Initialized to 0.0
        if self.armor_material and self.armor_thickness_mm > 0:
            # Assuming armor is over shield if shield exists, else over sheath
            inner_radius_armor = max(self.r_sheath_outer_m, self.r_shield_outer_m)
            cost_armor = (np.pi * (self.r_armor_outer_m**2 - inner_radius_armor**2) * self.length *
                          self.materials[self.armor_material]['density_kg_m3'] *
                          self.materials[self.armor_material]['cost_per_kg_usd'])
            
        total_cost = cost_conductor + cost_insulation + cost_sheath + cost_shield + cost_armor
        
        return {
            'conductor_cost_usd': cost_conductor,
            'insulation_cost_usd': cost_insulation,
            'sheath_cost_usd': cost_sheath,
            'shield_cost_usd': cost_shield,
            'armor_cost_usd': cost_armor, # Corrected: used 'cost_armor'
            'total_material_cost_usd': total_cost
        }

    def _calculate_sccr(self):
        # Simplified SCCR check based on conductor material property
        # In reality, SCCR depends on fault duration, conductor size, and system impedance.
        # Here, we compare the rated SCCR from material properties with a given system SCCR.
        # This is a placeholder for a more complex calculation.
        conductor_sccr_A = self.materials[self.conductor_material].get('sccr_A', 0)
        
        if conductor_sccr_A >= self.sccr_kA * 1000:
            return f"OK (Cable rated for {conductor_sccr_A/1000:.1f} kA, System SCCR: {self.sccr_kA:.1f} kA)"
        else:
            return f"FAIL (Cable rated for {conductor_sccr_A/1000:.1f} kA, System SCCR: {self.sccr_kA:.1f} kA)"

    def _calculate_bending_radius(self):
        # Minimum bending radius check
        # For single core cables, typically 12-20 times overall diameter
        min_bending_radius_required_m = self.overall_diameter_m * self.min_bending_radius_multiplier
        
        # This is a check against an implied installation minimum bending radius.
        # Without an explicit 'installed_bending_radius' parameter, we can only report the requirement.
        return f"Minimum required bending radius: {min_bending_radius_required_m * 1000:.2f} mm (based on {self.min_bending_radius_multiplier}x OD)"

    def _calculate_mechanical_properties(self):
        """
        Calculates basic mechanical properties like max tensile and compressive force capacity.
        This is a simplified model based on material yield strength and cross-sectional area.
        Actual mechanical analysis (e.g., for pulling tension, crushing) is more complex.
        """
        A_conductor_m2 = np.pi * (self.r_conductor_m**2)

        cond_props = self.materials[self.conductor_material]
        # Use yield strength for capacity calculation as it's the limit for elastic deformation
        yield_strength_cond_Pa = cond_props.get('yield_strength_MPa', 0) * 1e6

        # Max tensile force for conductor (N) - sum over all conductors
        max_tensile_force_conductor_N = yield_strength_cond_Pa * A_conductor_m2 * self.num_conductors

        # Max compressive force for conductor (N)
        max_compressive_force_conductor_N = yield_strength_cond_Pa * A_conductor_m2 * self.num_conductors

        A_armor_m2 = 0.0
        max_tensile_force_armor_N = 0.0
        max_compressive_force_armor_N = 0.0

        if self.armor_material and self.armor_thickness_mm > 0:
            armor_props = self.materials[self.armor_material]
            inner_radius_armor = max(self.r_sheath_outer_m, self.r_shield_outer_m)
            A_armor_m2 = np.pi * (self.r_armor_outer_m**2 - inner_radius_armor**2)
            yield_strength_armor_Pa = armor_props.get('yield_strength_MPa', 0) * 1e6
            max_tensile_force_armor_N = yield_strength_armor_Pa * A_armor_m2
            max_compressive_force_armor_N = yield_strength_armor_Pa * A_armor_m2 # Simplified
        
        # Total cable mechanical strength (simplified sum of conductor and armor contribution)
        total_max_tensile_N = max_tensile_force_conductor_N + max_tensile_force_armor_N
        total_max_compressive_N = max_compressive_force_conductor_N + max_compressive_force_armor_N # Corrected this line

        return {
            'conductor_cross_sectional_area_m2': A_conductor_m2 * self.num_conductors,
            'armor_cross_sectional_area_m2': A_armor_m2,
            'max_tensile_force_N': total_max_tensile_N,
            'max_compressive_force_N': total_max_compressive_N
        }

    def _calculate_localized_termination_temp_rise(self, termination_loss_W_per_termination, ambient_temp_C):
        """
        Estimates the localized temperature rise in the cable due to termination losses.
        This is a highly simplified conceptual model. For a precise analysis, detailed thermal modeling (e.g., Finite Element Analysis) of the termination area is required.
        Assumes heat from each termination spreads into a fixed 'influence length' of the cable.
        """
        # Total heat generated by all terminations
        total_termination_heat_W = self.num_terminations * self.num_conductors * termination_loss_W_per_termination

        # Define a conceptual length over which the termination heat dissipates into the cable
        # This length is a critical parameter and would ideally come from experimental data or detailed simulations.
        termination_influence_length_m = 10.0 # meters, a plausible distance over which termination heat affects the cable body

        if termination_influence_length_m <= 0:
            return {
                'estimated_local_temp_rise_C': 0.0,
                'affected_length_m': 0.0,
                'total_termination_heat_W': total_termination_heat_W,
                'note': 'Termination influence length is zero or negative.'
            }

        # Calculate an equivalent distributed heat per meter over the influence length
        equivalent_distributed_heat_W_per_m = total_termination_heat_W / termination_influence_length_m

        # Estimate the cable's thermal resistance (using a placeholder sheath_temp for calculation)
        # The exact sheath temperature doesn't critically affect thermal resistance for a general estimate here.
        R_thermal_cable_avg = self.calculate_thermal_resistance(
            ambient_temp_C + 10, # Assumed sheath temp slightly above ambient
            ambient_temp_C,
            burial_depth_m=self.burial_depth_m if self.laying_method == 'direct_buried' else None
        )

        # Localized temperature rise due to terminations spreading into the cable body
        temp_rise_due_to_terminations_C = equivalent_distributed_heat_W_per_m * R_thermal_cable_avg

        return {
            'estimated_local_temp_rise_C': temp_rise_due_to_terminations_C,
            'affected_length_m': termination_influence_length_m,
            'total_termination_heat_W': total_termination_heat_W,
            'note': 'Simplified model; for precise results, detailed thermal analysis of termination is needed.'
        }


    def _run_compliance_checks(self, current_A, mechanical_results):
        compliance = {
            'bending_radius': self._calculate_bending_radius(),
            'sccr': self._calculate_sccr(),
            'max_operating_temp_conductor': "SKIPPED", # Requires actual temp calculation...
            'max_tensile_capacity_N': f"{mechanical_results['max_tensile_force_N']:.2f} N (based on conductor and armor yield strength)",
            'max_compressive_capacity_N': f"{mechanical_results['max_compressive_force_N']:.2f} N (based on conductor and armor yield strength)"
        }
        return compliance

    def calculate_emi_shielding_effectiveness(self, frequency_hz=1e6): # Default to 1 MHz for EMI
        """
        Calculates the EMI shielding effectiveness (SE) of the cable.
        This is a simplified model, primarily considering absorption and reflection losses for a shielded cable.
        Detailed EMI analysis requires complex electromagnetic simulations.
        Reference: Typically based on Schelkunoff's theory for planar shields, adapted for cylindrical geometry.
        
        SE (dB) = A (Absorption Loss) + R (Reflection Loss) + B (Multiple Reflection Correction)
        For thick shields or high frequencies, B approaches 0 and can often be ignored.
        """
        if not self.shield_material or self.shield_thickness_mm == 0:
            return {'Error': 'No shield material or zero shield thickness. Cannot calculate EMI shielding effectiveness.'}

        if not self.r_shield_outer_m or self.r_shield_outer_m <= self.r_sheath_outer_m:
             return {'Error': 'Shield outer radius is not defined or is not greater than sheath outer radius. Cannot calculate EMI shielding effectiveness.'}
        
        shield_props = self.materials[self.shield_material]
        conductivity_S_m = shield_props.get('conductivity_S_m')
        permeability_relative = shield_props.get('permeability_relative')
        shield_thickness_m = self.shield_thickness_mm / 1000.0 # Convert to meters

        if not conductivity_S_m or not permeability_relative:
            return {'Error': f"Missing conductivity_S_m or permeability_relative for shield material '{self.shield_material}'."}

        mu_0 = 4 * np.pi * 1e-7 # Permeability of free space (H/m)
        mu = permeability_relative * mu_0 # Absolute permeability of shield material

        # Skin depth (delta)
        # delta = sqrt(2 / (omega * mu * sigma))
        omega = 2 * np.pi * frequency_hz
        try:
            skin_depth_m = np.sqrt(2 / (omega * mu * conductivity_S_m))
        except ZeroDivisionError:
            return {'Error': 'Shield material conductivity is zero. Cannot calculate skin depth.'}

        # Absorption Loss (A) - in dB
        # A = 8.686 * (shield_thickness_m / skin_depth_m)
        absorption_loss_dB = 8.686 * (shield_thickness_m / skin_depth_m)

        # Wave impedance of free space (eta_0) and shield material (eta_s)
        eta_0 = 377 # Ohms (approx for free space)
        try:
            eta_s = np.sqrt(omega * mu / conductivity_S_m)
        except ZeroDivisionError:
            return {'Error': 'Shield material conductivity is zero. Cannot calculate wave impedance.'}

        # Reflection Loss (R) - simplified for far-field plane wave incidence (dB)
        # For cylindrical shields, it's more complex, but this is a common simplification.
        # R = 20 * log10( |eta_0 + eta_s|^2 / (4 * eta_0 * eta_s) ) approx. 20 * log10(eta_0 / (4 * eta_s)) for eta_0 >> eta_s
        reflection_loss_dB = 20 * np.log10(eta_0 / (4 * eta_s)) # Simplified

        # Multiple Reflection Correction (B) - often negligible if A > 10 dB
        # For simplicity, we'll often ignore B for power cables or assume it's small if absorption is high.
        # B = 20 * log10( |1 - ((eta_s - eta_0)/(eta_s + eta_0))^2 * e^(-2*t/delta)| )
        # If A > 10dB, B is typically small.

        total_se_dB = absorption_loss_dB + reflection_loss_dB
        note = None
        if absorption_loss_dB < 10:
            note = "Absorption loss is low; multiple reflection correction (B) might be significant and is not included in this simplified calculation."
        elif self.shield_coverage_percentage < 1.0:
            total_se_dB *= self.shield_coverage_percentage # Reduce SE for incomplete coverage
            note = f"Shielding Effectiveness reduced due to {self.shield_coverage_percentage*100:.1f}% coverage."


        return {
            'Total SE': total_se_dB,
            'Absorption Loss': absorption_loss_dB,
            'Reflection Loss': reflection_loss_dB,
            'Note': note
        }

    def generate_markdown_report(self, results, filename="cable_design_report.md"):
        """
        Generates a detailed markdown report of the cable design and calculation results.
        """
        report_content = f"# Cable Design Report\n\n"
        report_content += "## 1. Cable Parameters\n"
        report_content += f"- Conductor Material: {self.conductor_material}\n"
        report_content += f"- Conductor Diameter: {self.conductor_diameter_mm} mm\n"
        report_content += f"- Insulation Material: {self.insulation_material}\n"
        report_content += f"- Insulation Thickness: {self.insulation_thickness_mm} mm\n"
        report_content += f"- Sheath Material: {self.sheath_material}\n"
        report_content += f"- Sheath Thickness: {self.sheath_thickness_mm} mm\n"
        if self.shield_material:
            report_content += f"- Shield Material: {self.shield_material}\n"
            report_content += f"- Shield Thickness: {self.shield_thickness_mm} mm\n"
            report_content += f"- Shield Type: {self.shield_type}\n"
            report_content += f"- Shield Coverage: {self.shield_coverage_percentage*100:.1f}%\n"
            report_content += f"- Shield Bonding: {self.shield_bonding}\n"
        if self.armor_material:
            report_content += f"- Armor Material: {self.armor_material}\n"
            report_content += f"- Armor Thickness: {self.armor_thickness_mm} mm\n"
        report_content += f"- Operating Voltage: {self.operating_voltage_kV} kV\n"
        report_content += f"- Operating Frequency: {self.operating_frequency_hz} Hz\n"
        report_content += f"- Number of Conductors: {self.num_conductors}\n"
        report_content += f"- Cable Length: {self.length_km} km\n"
        report_content += f"- Laying Method: {self.laying_method}\n"
        if self.laying_method == 'direct_buried':
            report_content += f"- Burial Depth: {self.burial_depth_m} m\n"
            report_content += f"- Thermal Resistivity of Soil: {self.thermal_resistivity_soil_Km_W} K.m/W\n"
            report_content += f"- Ground Temperature: {self.ground_temp_C} °C\n"
        elif self.laying_method == 'in_air':
            report_content += f"- Ambient Air Temperature: {self.ambient_air_temp_C} °C\n"
            report_content += f"- Solar Radiation: {self.solar_radiation_W_m2} W/m²\n"
            report_content += f"- Wind Speed: {self.wind_speed_m_s} m/s\n"
        report_content += f"- Number of Terminations: {self.num_terminations}\n"
        report_content += f"- Loss per Termination: {self.termination_loss_per_termination_W} W\n"
        report_content += f"- System SCCR: {self.sccr_kA} kA\n"
        report_content += f"- Installation Temperature: {self.installation_temp_C} °C\n"
        report_content += f"- Min Bending Radius Multiplier: {self.min_bending_radius_multiplier}x OD\n"
        report_content += f"- Load Factor: {self.load_factor}\n"
        report_content += f"- Losses Percentage: {self.losses_pct}%\n"

        report_content += "\n## 2. Geometric Properties\n"
        report_content += f"- Conductor Radius: {self.r_conductor_m * 1000:.2f} mm\n"
        report_content += f"- Insulation Outer Radius: {self.r_insulation_outer_m * 1000:.2f} mm\n"
        report_content += f"- Sheath Outer Radius: {self.r_sheath_outer_m * 1000:.2f} mm\n"
        if self.shield_material and self.shield_thickness_mm > 0:
            report_content += f"- Shield Outer Radius: {self.r_shield_outer_m * 1000:.2f} mm\n"
        if self.armor_material and self.armor_thickness_mm > 0:
            report_content += f"- Armor Outer Radius: {self.r_armor_outer_m * 1000:.2f} mm\n"
        report_content += f"- Overall Diameter: {self.overall_diameter_m * 1000:.2f} mm\n"

        report_content += "\n## 3. Calculation Results\n"
        if 'ampacity_results' in results and 'ampacity_A' in results['ampacity_results']:
            report_content += "### Ampacity and Thermal Performance\n"
            report_content += f"- Calculated Ampacity (I_z): {results['ampacity_results']['ampacity_A']:.2f} A\n"
            report_content += f"- Conductor Temperature at Ampacity: {results['ampacity_results']['conductor_temp_at_ampacity']:.2f} °C\n"
            report_content += f"- Total Heat Dissipated per meter: {results['ampacity_results']['total_heat_dissipated_W_per_m']:.2f} W/m\n"
            report_content += f"- Total Thermal Resistance: {results['ampacity_results']['thermal_resistance_total_Km_W']:.2f} K.m/W\n"
            report_content += f"- AC Resistance per meter: {results['ampacity_results']['resistance_ac_ohm_m']:.6f} Ohm/m\n"
            report_content += f"- Dielectric Losses per meter: {results['ampacity_results']['dielectric_losses_W_per_m']:.2f} W/m\n"
            report_content += f"- Shield Losses per meter: {results['ampacity_results']['shield_losses_W_per_m']:.2f} W/m\n"
            report_content += f"- Termination Heat per km: {results['ampacity_results']['termination_heat_W_per_km']:.2f} W/km\n"
        else:
            report_content += "- Ampacity results not available or calculated as 0.\n"

        if 'voltage_drop_V' in results:
            report_content += "\n### Voltage Drop\n"
            report_content += f"- Total Voltage Drop over {self.length_km} km: {results['voltage_drop_V']:.2f} V\n"
            report_content += f"- Percentage Voltage Drop: {results['voltage_drop_percentage']:.2f} %\n"

        if 'cost_results' in results:
            report_content += "\n### Cost Analysis\n"
            report_content += f"- Total Material Cost: ${results['cost_results']['total_material_cost_usd']:.2f}\n"
            report_content += f"  - Conductor Cost: ${results['cost_results']['conductor_cost_usd']:.2f}\n"
            report_content += f"  - Insulation Cost: ${results['cost_results']['insulation_cost_usd']:.2f}\n"
            report_content += f"  - Sheath Cost: ${results['cost_results']['sheath_cost_usd']:.2f}\n"
            if self.shield_material:
                report_content += f"  - Shield Cost: ${results['cost_results']['shield_cost_usd']:.2f}\n"
            if self.armor_material:
                report_content += f"  - Armor Cost: ${results['cost_results']['armor_cost_usd']:.2f}\n"
        
        report_content += "\n### Compliance Checks\n"
        if 'compliance_checks' in results:
            for check, status in results['compliance_checks'].items():
                report_content += f"- {check.replace('_', ' ').title()}: {status}\n"

        if 'emi_shielding_effectiveness' in results:
            report_content += "\n### EMI Shielding Effectiveness\n"
            emi_results = results['emi_shielding_effectiveness']
            if 'Error' in emi_results:
                report_content += f"- Error: {emi_results['Error']}\n"
            else:
                report_content += f"- Total Shielding Effectiveness (SE): {emi_results['Total SE']:.2f} dB\n"
                report_content += f"  - Absorption Loss (A): {emi_results['Absorption Loss']:.2f} dB\n"
                report_content += f"  - Reflection Loss (R): {emi_results['Reflection Loss']:.2f} dB\n"
                if 'Note' in emi_results:
                    report_content += f"  - Note: {emi_results['Note']}\n"
        
        if 'localized_termination_temp_rise' in results and 'estimated_local_temp_rise_C' in results['localized_termination_temp_rise']:
            report_content += "\n### Localized Termination Temperature Rise (Simplified Model)\n"
            term_temp_results = results['localized_termination_temp_rise']
            report_content += f"- Estimated Local Temp Rise at Termination: {term_temp_results['estimated_local_temp_rise_C']:.2f} °C\n"
            report_content += f"- Affected Length: {term_temp_results['affected_length_m']:.2f} m\n"
            report_content += f"- Total Termination Heat: {term_temp_results['total_termination_heat_W']:.2f} W\n"
            if 'note' in term_temp_results:
                report_content += f"- Note: {term_temp_results['note']}\n"


        try:
            with open(filename, 'w') as f:
                f.write(report_content)
            print(f"Report generated successfully: {filename}")
        except IOError as e:
            print(f"Error writing report to file {filename}: {e}")

    def _plot_geometric_dimensions(self, results, example_name):
        """
        Plots the geometric dimensions of the cable cross-section.
        """
        plt.figure(figsize=(6, 6))
        ax = plt.gca()

        # Conductor
        conductor_circle = plt.Circle((0, 0), self.r_conductor_m * 1000, color='red', label='Conductor')
        ax.add_patch(conductor_circle)

        # Insulation
        insulation_circle = plt.Circle((0, 0), self.r_insulation_outer_m * 1000, color='lightgray', fill=False, linestyle='--', label='Insulation')
        ax.add_patch(insulation_circle)

        # Sheath
        sheath_circle = plt.Circle((0, 0), self.r_sheath_outer_m * 1000, color='darkgray', fill=False, linestyle=':', label='Sheath')
        ax.add_patch(sheath_circle)

        # Shield (if present)
        if self.shield_material and self.shield_thickness_mm > 0:
            shield_circle = plt.Circle((0, 0), self.r_shield_outer_m * 1000, color='blue', fill=False, linestyle='-.', label='Shield')
            ax.add_patch(shield_circle)

        # Armor (if present)
        if self.armor_material and self.armor_thickness_mm > 0:
            armor_circle = plt.Circle((0, 0), self.r_armor_outer_m * 1000, color='green', fill=False, linestyle='-', label='Armor')
            ax.add_patch(armor_circle)

        ax.set_xlim(-(self.overall_diameter_m / 2) * 1000 * 1.2, (self.overall_diameter_m / 2) * 1000 * 1.2)
        ax.set_ylim(-(self.overall_diameter_m / 2) * 1000 * 1.2, (self.overall_diameter_m / 2) * 1000 * 1.2)

        plt.title(f"Geometric Dimensions{example_name}")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
        plt.tight_layout()
        
        # Construct a clean filename for saving
        filename_base = f"geometric_dimensions{example_name.replace(' - ', '_').replace(' ', '_').replace('(', '').replace(')', '').lower()}"
        plt.savefig(f"{filename_base}.png")
        plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    materials_data = load_material_properties(MATERIALS_FILE_PATH)

    if materials_data:
        # Example 1: Standard Single Core Cable
        print("\\n--- Running Example 1: Standard Single Core Cable ---")
        try:
            cable1 = CableDesigner(
                materials_data=materials_data,
                conductor_material='Copper',
                conductor_diameter_mm=15.0,
                insulation_material='XLPE',
                insulation_thickness_mm=5.0,
                sheath_material='PVC',
                sheath_thickness_mm=2.0,
                operating_voltage_kV=11.0,
                operating_frequency_hz=50.0,
                num_conductors=1,
                burial_depth_m=0.8,
                thermal_resistivity_soil_Km_W=1.0,
                ground_temp_C=20.0,
                laying_method='direct_buried',
                length_km=1.0 # Added length_km parameter
            )
            
            # Calculate ampacity at max operating temperature for conductor material
            conductor_max_temp_C = materials_data['Copper']['max_operating_temp_C']
            results_1 = cable1.calculate_current_capacity(
                conductor_temp_C=conductor_max_temp_C,
                ambient_temp_C=25.0, # Ambient ground temp
                length_km=cable1.length_km,
                burial_depth_m=cable1.burial_depth_m,
                ground_temp_C=cable1.ground_temp_C,
                apply_termination_effect=True # Apply termination effect
            )

            # Recalculate voltage drop for a nominal current (e.g., 80% of ampacity)
            nominal_current_1 = results_1['ampacity_results']['ampacity_A'] * 0.8
            voltage_drop_V_1 = cable1.calculate_voltage_drop(nominal_current_1)
            # Assuming line-to-line voltage for percentage calculation
            voltage_drop_percentage_1 = (voltage_drop_V_1 / (cable1.operating_voltage_kV * 1000 * np.sqrt(3))) * 100 # For 3-phase, divide by Line-to-Neutral voltage if Vdrop is phase-to-neutral, or by Line-to-Line if Vdrop is line-to-line

            # Calculate cost
            cost_results_1 = cable1.calculate_cost()

            # Calculate localized termination temp rise (for reporting)
            localized_termination_temp_rise_1 = cable1._calculate_localized_termination_temp_rise(
                cable1.termination_loss_per_termination_W,
                cable1.ambient_air_temp_C # Using ambient air temp as reference for termination area
            )
            
            # Update results_1 with additional calculations
            results_1['voltage_drop_V'] = voltage_drop_V_1
            results_1['voltage_drop_percentage'] = voltage_drop_percentage_1
            results_1['cost_results'] = cost_results_1
            results_1['localized_termination_temp_rise'] = localized_termination_temp_rise_1
            results_1['emi_shielding_effectiveness'] = cable1.calculate_emi_shielding_effectiveness()

            # Print results for Example 1
            print("\\n--- Calculation Results (Example 1) ---")
            if 'ampacity_results' in results_1 and 'ampacity_A' in results_1['ampacity_results']:
                print(f"Calculated Ampacity: {results_1['ampacity_results']['ampacity_A']:.2f} A")
                print(f"Total Voltage Drop: {results_1['voltage_drop_V']:.2f} V ({results_1['voltage_drop_percentage']:.2f} %)")
                print(f"Total Material Cost: ${results_1['cost_results']['total_material_cost_usd']:.2f}")
                print(f"Termination Heat per km: {results_1['ampacity_results']['termination_heat_W_per_km']:.2f} W/km")

            print("\\n--- Compliance Checks (Example 1) ---")
            print(f"Bending Radius Check: {results_1['compliance_checks']['bending_radius']}")
            print(f"SCCR Check: {results_1['compliance_checks']['sccr']}")
            print(f"Max Tensile Capacity: {results_1['compliance_checks']['max_tensile_capacity_N']}")
            print(f"Max Compressive Capacity: {results_1['compliance_checks']['max_compressive_capacity_N']}")

            print("\\n--- EMI Shielding Effectiveness (Example 1) ---")
            emi_results_1 = results_1['emi_shielding_effectiveness']
            if 'Error' in emi_results_1:
                print(f"EMI Calculation Error: {emi_results_1['Error']}")
            else:
                print(f"Total Shielding Effectiveness (SE): {emi_results_1['Total SE']:.2f} dB")
                print(f"  Absorption Loss (A): {emi_results_1['Absorption Loss']:.2f} dB")
                print(f"  Reflection Loss (R): {emi_results_1['Reflection Loss']:.2f} dB")
                if 'Note' in emi_results_1:
                    print(f"  Note: {emi_results_1['Note']}")

            # Generate detailed markdown report for Example 1
            cable1.generate_markdown_report(results_1, "cable_design_report_example1.md")

            # Plot geometric dimensions for Example 1
            cable1._plot_geometric_dimensions(results_1, " - Example 1 (Standard Cable)")


        except ValueError as ve:
            print(f"Error in Example 1: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred in Example 1: {e}")

        # Example 2: Armored Cable with Shield
        print("\\n--- Running Example 2: Armored Cable with Shield ---")
        try:
            cable2 = CableDesigner(
                materials_data=materials_data,
                conductor_material='Aluminum',
                conductor_diameter_mm=25.0,
                insulation_material='XLPE',
                insulation_thickness_mm=8.0,
                sheath_material='PE', # Corrected from HDPE to PE
                sheath_thickness_mm=3.0,
                shield_material='Copper', # Added shield
                shield_thickness_mm=0.5,
                shield_type='tape', # Example shield type
                shield_coverage_percentage=90.0,
                armor_material='Steel', # Added armor
                armor_thickness_mm=4.0,
                operating_voltage_kV=33.0,
                operating_frequency_hz=50.0,
                num_conductors=3,
                laying_method='in_air',
                ambient_air_temp_C=35.0,
                solar_radiation_W_m2=900.0,
                wind_speed_m_s=1.0,
                length_km=2.5, # Longer cable
                shield_bonding='multi_point' # Different bonding for example
            )

            # Calculate ampacity at max operating temperature for conductor material
            conductor_max_temp_C_2 = materials_data['Aluminum']['max_operating_temp_C']
            results_2 = cable2.calculate_current_capacity(
                conductor_temp_C=conductor_max_temp_C_2,
                ambient_temp_C=cable2.ambient_air_temp_C, # Ambient air temp
                length_km=cable2.length_km,
                apply_termination_effect=True # Apply termination effect
            )

            # Recalculate voltage drop for a nominal current (e.g., 80% of ampacity)
            nominal_current_2 = results_2['ampacity_results']['ampacity_A'] * 0.8
            voltage_drop_V_2 = cable2.calculate_voltage_drop(nominal_current_2)
            # Assuming line-to-line voltage for percentage calculation
            voltage_drop_percentage_2 = (voltage_drop_V_2 / (cable2.operating_voltage_kV * 1000 * np.sqrt(3))) * 100

            # Calculate cost
            cost_results_2 = cable2.calculate_cost()

            # Calculate localized termination temp rise (for reporting)
            localized_termination_temp_rise_2 = cable2._calculate_localized_termination_temp_rise(
                cable2.termination_loss_per_termination_W,
                cable2.ambient_air_temp_C # Using ambient air temp as reference for termination area
            )

            # Update results_2 with additional calculations
            results_2['voltage_drop_V'] = voltage_drop_V_2
            results_2['voltage_drop_percentage'] = voltage_drop_percentage_2
            results_2['cost_results'] = cost_results_2
            results_2['localized_termination_temp_rise'] = localized_termination_temp_rise_2
            results_2['emi_shielding_effectiveness'] = cable2.calculate_emi_shielding_effectiveness()


            # Print results for Example 2
            print("\\n--- Calculation Results (Example 2) ---")
            if 'ampacity_results' in results_2 and 'ampacity_A' in results_2['ampacity_results']:
                print(f"Calculated Ampacity: {results_2['ampacity_results']['ampacity_A']:.2f} A")
                print(f"Total Voltage Drop: {results_2['voltage_drop_V']:.2f} V ({results_2['voltage_drop_percentage']:.2f} %)")
                print(f"Total Material Cost: ${results_2['cost_results']['total_material_cost_usd']:.2f}")
                print(f"Termination Heat per km: {results_2['ampacity_results']['termination_heat_W_per_km']:.2f} W/km")

            print("\\n--- Compliance Checks (Example 2) ---")
            print(f"Bending Radius Check: {results_2['compliance_checks']['bending_radius']}")
            print(f"SCCR Check: {results_2['compliance_checks']['sccr']}")
            print(f"Max Tensile Capacity: {results_2['compliance_checks']['max_tensile_capacity_N']}")
            print(f"Max Compressive Capacity: {results_2['compliance_checks']['max_compressive_capacity_N']}")

            print("\\n--- EMI Shielding Effectiveness (Example 2) ---")
            emi_results_2 = results_2['emi_shielding_effectiveness']
            if 'Error' in emi_results_2:
                print(f"EMI Calculation Error: {emi_results_2['Error']}")
            else:
                print(f"Total Shielding Effectiveness (SE): {emi_results_2['Total SE']:.2f} dB")
                print(f"  Absorption Loss (A): {emi_results_2['Absorption Loss']:.2f} dB")
                print(f"  Reflection Loss (R): {emi_results_2['Reflection Loss']:.2f} dB")
                if 'Note' in emi_results_2:
                    print(f"  Note: {emi_results_2['Note']}")

            # Generate detailed markdown report for Example 2
            cable2.generate_markdown_report(results_2, "cable_design_report_example2.md")

            # Plot geometric dimensions for Example 2
            cable2._plot_geometric_dimensions(results_2, " - Example 2 (Armored Cable)")

        except ValueError as ve:
            print(f"Error in Example 2: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred in Example 2: {e}")