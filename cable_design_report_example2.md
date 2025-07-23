# Cable Design Report

## 1. Cable Parameters
- Conductor Material: Aluminum
- Conductor Diameter: 25.0 mm
- Insulation Material: XLPE
- Insulation Thickness: 8.0 mm
- Sheath Material: PE
- Sheath Thickness: 3.0 mm
- Shield Material: Copper
- Shield Thickness: 0.5 mm
- Shield Type: tape
- Shield Coverage: 90.0%
- Shield Bonding: multi_point
- Armor Material: Steel
- Armor Thickness: 4.0 mm
- Operating Voltage: 33.0 kV
- Operating Frequency: 50.0 Hz
- Number of Conductors: 3
- Cable Length: 2.5 km
- Laying Method: in_air
- Ambient Air Temperature: 35.0 °C
- Solar Radiation: 900.0 W/m²
- Wind Speed: 1.0 m/s
- Number of Terminations: 2
- Loss per Termination: 10.0 W
- System SCCR: 50 kA
- Installation Temperature: 25.0 °C
- Min Bending Radius Multiplier: 12x OD
- Load Factor: 1.0
- Losses Percentage: 5.0%

## 2. Geometric Properties
- Conductor Radius: 12.50 mm
- Insulation Outer Radius: 20.50 mm
- Sheath Outer Radius: 23.50 mm
- Shield Outer Radius: 24.00 mm
- Armor Outer Radius: 28.00 mm
- Overall Diameter: 56.00 mm

## 3. Calculation Results
### Ampacity and Thermal Performance
- Calculated Ampacity (I_z): 391.22 A
- Conductor Temperature at Ampacity: 90.00 °C
- Total Heat Dissipated per meter: 41.51 W/m
- Total Thermal Resistance: 1.33 K.m/W
- AC Resistance per meter: 0.000074 Ohm/m
- Dielectric Losses per meter: 0.08 W/m
- Shield Losses per meter: 7.61 W/m
- Termination Heat per km: 24.00 W/km

### Voltage Drop
- Total Voltage Drop over 2.5 km: 229.68 V
- Percentage Voltage Drop: 0.40 %

### Cost Analysis
- Total Material Cost: $67618.25
  - Conductor Cost: $24850.49
  - Insulation Cost: $17168.18
  - Sheath Cost: $1969.78
  - Shield Cost: $13370.62
  - Armor Cost: $10259.18

### Compliance Checks
- Bending Radius: Minimum required bending radius: 672.00 mm (based on 12x OD)
- Sccr: OK (Cable rated for 65.0 kA, System SCCR: 50.0 kA)
- Max Operating Temp Conductor: SKIPPED
- Max Tensile Capacity N: 214904.57 N (based on conductor and armor yield strength)
- Max Compressive Capacity N: 214904.57 N (based on conductor and armor yield strength)

### EMI Shielding Effectiveness
- Total Shielding Effectiveness (SE): 156.48 dB
  - Absorption Loss (A): 65.72 dB
  - Reflection Loss (R): 108.15 dB
  - Note: Shielding Effectiveness reduced due to 90.0% coverage.

### Localized Termination Temperature Rise (Simplified Model)
- Estimated Local Temp Rise at Termination: 7.95 °C
- Affected Length: 10.00 m
- Total Termination Heat: 60.00 W
- Note: Simplified model; for precise results, detailed thermal analysis of termination is needed.
