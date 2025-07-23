# Cable Design Report

## 1. Cable Parameters
- Conductor Material: Copper
- Conductor Diameter: 15.0 mm
- Insulation Material: XLPE
- Insulation Thickness: 5.0 mm
- Sheath Material: PVC
- Sheath Thickness: 2.0 mm
- Operating Voltage: 11.0 kV
- Operating Frequency: 50.0 Hz
- Number of Conductors: 1
- Cable Length: 1.0 km
- Laying Method: direct_buried
- Burial Depth: 0.8 m
- Thermal Resistivity of Soil: 1.0 K.m/W
- Ground Temperature: 20.0 °C
- Number of Terminations: 2
- Loss per Termination: 10.0 W
- System SCCR: 50 kA
- Installation Temperature: 25.0 °C
- Min Bending Radius Multiplier: 12x OD
- Load Factor: 1.0
- Losses Percentage: 5.0%

## 2. Geometric Properties
- Conductor Radius: 7.50 mm
- Insulation Outer Radius: 12.50 mm
- Sheath Outer Radius: 14.50 mm
- Overall Diameter: 29.00 mm

## 3. Calculation Results
### Ampacity and Thermal Performance
- Calculated Ampacity (I_z): 667.03 A
- Conductor Temperature at Ampacity: 90.00 °C
- Total Heat Dissipated per meter: 55.22 W/m
- Total Thermal Resistance: 1.18 K.m/W
- AC Resistance per meter: 0.000124 Ohm/m
- Dielectric Losses per meter: 0.00 W/m
- Shield Losses per meter: 0.00 W/m
- Termination Heat per km: 20.00 W/km

### Voltage Drop
- Total Voltage Drop over 1.0 km: 76.55 V
- Percentage Voltage Drop: 0.40 %

### Cost Analysis
- Total Material Cost: $13890.24
  - Conductor Cost: $12666.90
  - Insulation Cost: $867.08
  - Sheath Cost: $356.26

### Compliance Checks
- Bending Radius: Minimum required bending radius: 348.00 mm (based on 12x OD)
- Sccr: OK (Cable rated for 65.0 kA, System SCCR: 50.0 kA)
- Max Operating Temp Conductor: SKIPPED
- Max Tensile Capacity N: 12370.02 N (based on conductor and armor yield strength)
- Max Compressive Capacity N: 12370.02 N (based on conductor and armor yield strength)

### EMI Shielding Effectiveness
- Error: No shield material or zero shield thickness. Cannot calculate EMI shielding effectiveness.

### Localized Termination Temperature Rise (Simplified Model)
- Estimated Local Temp Rise at Termination: 2.35 °C
- Affected Length: 10.00 m
- Total Termination Heat: 20.00 W
- Note: Simplified model; for precise results, detailed thermal analysis of termination is needed.
