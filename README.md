# Advanced Cable Design & Analysis Tool

## Overview

This repository presents a robust and versatile Python-based tool designed for the comprehensive **design, analysis, and optimization of power cables**. Leveraging fundamental electrical, thermal, and mechanical engineering principles, the tool automates complex calculations and integrates with Computer-Aided Design (CAD) software, providing a powerful, cost-effective, and fully open-source solution for cable engineering.

Developed as a showcase of advanced engineering capabilities and a deep understanding of cable technology, this project demonstrates a highly practical approach to real-world challenges in cable manufacturing and design, making it ideal for roles in leading companies within the energy or telecommunications sectors.

## Key Features

* **Comprehensive Electrical Analysis:** Calculation of AC/DC resistance, inductance, capacitance, dielectric losses, and voltage drop.
* **Thermal Performance Modeling:** Ampacity calculations based on IEC 60287 standards, considering various laying methods and environmental conditions.
* **Mechanical Integrity Assessment:** Evaluation of bending radius, tensile strength, and compressive capacity.
* **Electromagnetic Interference (EMI) Shielding Analysis:** Calculation of shielding effectiveness (SE) for various shield configurations using modern RF engineering principles.
* **Material Property Management:** Flexible loading and utilization of material properties from a JSON database (`materials.json`).
* **Automated Reporting:** Generation of detailed, human-readable Markdown reports summarizing all design parameters, calculation results, and compliance checks.
* **3D Geometric Modeling Integration:** Direct integration with **FreeCAD** (a powerful open-source CAD software) to generate precise 3D cross-sectional models of designed cables, facilitating visual inspection and manufacturing guidance.
* **Cost Estimation:** Basic material cost estimation based on design and material properties.
* **Compliance Checks:** Automated checks against predefined criteria (e.g., Short Circuit Current Rating - SCCR, minimum bending radius).

## Alignment with Cable Design Engineer Role Requirements

This project directly addresses and demonstrates proficiency in the vast majority of the responsibilities and qualifications typically required for a **Cable Design Engineer** position in a leading manufacturing environment. It showcases how a deep understanding of cable engineering principles, combined with modern computational and **open-source tools**, can meet the demanding requirements of this specialized field.

**This project confirms the feasibility of achieving 100% of the technical requirements typically outlined for such roles, utilizing readily available, free, and open-source tools.**

---

### **Demonstrated Responsibilities:**

* **"Design and develop cable systems according to project requirements, considering factors such as electrical, mechanical, and environmental constraints."**
    * **Project Alignment:** This tool is precisely designed for this purpose. It takes various input parameters (e.g., conductor size, insulation type, armor, laying method) and performs detailed calculations for electrical (e.g., ampacity, voltage drop), mechanical (e.g., bending radius, tensile strength), and environmental (e.g., soil thermal resistivity, ambient temperature) constraints. The `materials.json` file allows for flexible material selection and property management.

* **"Conduct thorough analysis and calculations to determine cable specifications, including conductor size, insulation materials, shielding, and termination methods."**
    * **Project Alignment:** The core of the `CableDesigner` class is dedicated to these analyses. It calculates current ratings (ampacity), inductance, capacitance, losses, short-circuit current capacity, and EMI shielding effectiveness based on detailed input specifications for each layer (conductor, insulation, sheath, armor, shield) and termination conditions.

* **"Collaborate closely with cross-functional teams, including electrical engineers, mechanical engineers, and manufacturing personnel..."**
    * **Project Alignment:** The tool's outputs are designed for effective collaboration. The generated **detailed Markdown reports** provide clear specifications for all stakeholders. Critically, the **FreeCAD 3D models** offer a precise visual and geometric representation that is invaluable for mechanical engineers and manufacturing teams, ensuring seamless integration and understanding of the cable's physical design.

* **"Create detailed technical drawings, schematics, and specifications using computer-aided design (CAD) tools to guide the manufacturing process."**
    * **Project Alignment:** The direct integration with **FreeCAD** fulfills the CAD requirement. The tool generates 3D cross-sections that serve as highly detailed geometric "drawings" to guide the manufacturing process. Furthermore, the **Markdown reports** function as comprehensive technical specifications for each cable design.

* **"Perform testing and validation of cable prototypes and final products, utilizing industry-standard testing methods and equipment."**
    * **Project Alignment:** While this tool is a **design and prediction engine** (not a physical testing lab), it is fundamental to the testing and validation process. It provides precise theoretical values (e.g., calculated ampacity, voltage drop, EMI shielding) against which physical test results can be compared. By accurately predicting performance, the tool significantly **reduces the need for costly physical prototypes** and **streamlines the validation phase**, by identifying potential issues early in the design cycle.

* **"Identify and resolve design issues and challenges by conducting root cause analysis and implementing appropriate design modifications."**
    * **Project Alignment:** The tool is built for rapid iteration and analysis. Engineers can quickly modify design parameters (e.g., conductor size, insulation thickness, material properties) and immediately observe the impact on performance metrics (e.g., ampacity, EMI, mechanical stress). This capability is crucial for efficient root cause analysis and rapid implementation of design modifications. Automated compliance checks further aid in identifying issues.

* **"Stay up-to-date with industry trends, standards, and regulations related to cable design, and incorporate them into the design process as necessary."**
    * **Project Alignment:** The calculations within the tool are based on established engineering principles and industry standards (e.g., IEC 60287 for ampacity calculations, electromagnetic theory for EMI). The modular structure of the code allows for straightforward updates and incorporation of new standards, materials, or calculation methodologies as industry trends evolve.

* **"Collaborate with suppliers and vendors to select and evaluate materials, components, and manufacturing processes, ensuring high quality and cost-effectiveness."**
    * **Project Alignment:** The `materials.json` file serves as a configurable database for material properties from various suppliers. This allows engineers to easily compare and evaluate different material options based on their electrical, thermal, mechanical, and cost attributes, directly supporting material selection for quality and cost-effectiveness.

* **"Support the production team during the manufacturing process, providing guidance and troubleshooting assistance when necessary."**
    * **Project Alignment:** The detailed design specifications in the Markdown reports and the precise 3D models from FreeCAD provide clear, unambiguous guidance to the production team. These outputs are essential resources for manufacturing planning and troubleshooting.

* **"Maintain accurate documentation of design processes, specifications, and test results, and provide timely reports to stakeholders."**
    * **Project Alignment:** The tool automates the generation of comprehensive **Markdown reports** for each design scenario. These reports meticulously document all input parameters, calculated results, and compliance checks, ensuring accurate, standardized, and timely documentation for all stakeholders. The well-structured and commented Python code also serves as excellent documentation of the design process logic.

---

### **Demonstrated Qualifications:**

* **"Master's degree in Electrical Engineering or a related field."**
    * **Project Alignment:** The depth and breadth of the engineering calculations (electrical, thermal, mechanical, electromagnetic) and the integration of diverse engineering software libraries (NumPy, SciPy, ElectricPy, scikit-rf) reflect the advanced theoretical understanding and analytical skills typically acquired through a Master's degree in Electrical Engineering.

* **"Proven experience in cable design, preferably in a relevant industry such as telecommunications, automotive, aerospace, or energy."**
    * **Project Alignment:** This project *is* a direct demonstration of proven, hands-on experience in cable design. The inclusion of complex features like ampacity, short-circuit current, and EMI shielding calculations, with specific examples (e.g., armored cable), directly showcases relevant experience applicable to the **energy industry**.

* **"Proficiency in computer-aided design (CAD) software and other relevant engineering tools."**
    * **Project Alignment:** Proficiency in CAD is directly demonstrated through the **FreeCAD integration**. The entire project showcases strong proficiency in a suite of relevant engineering tools, including **Python** as the primary programming language, alongside powerful libraries such as **NumPy, Matplotlib, Pandas, SciPy, ElectricPy, and scikit-rf**, all of which are industry-standard for data analysis, scientific computing, and engineering simulations.

* **"Strong analytical and problem-solving skills, with the ability to conduct detailed calculations and perform technical analysis."**
    * **Project Alignment:** The very nature of this project, which involves implementing and orchestrating complex mathematical models for cable behavior, is a testament to strong analytical and problem-solving skills. The detailed calculations for every aspect of cable performance serve as direct evidence of this qualification.

* **"Solid understanding of electrical and mechanical principles and their application to cable design."**
    * **Project Alignment:** This understanding is fundamental to every line of code. The correct application of formulas for AC/DC resistance, inductance, capacitance, dielectric losses, thermal resistances (internal, insulation, external), ampacity, bending radius, tensile strength, compressive capacity, and electromagnetic shielding clearly demonstrates a solid grasp of both electrical and mechanical engineering principles as applied specifically to cable design.

* **"Excellent communication skills, both written and verbal, with the ability to convey complex technical concepts to diverse stakeholders."**
    * **Project Alignment:** The automatically generated, clear, and comprehensive Markdown reports (written communication) demonstrate the ability to document and convey complex technical concepts effectively. The structured nature of the Python code and the potential for a live demonstration of the tool (verbal communication) further illustrate this capability.

* **"Strong attention to detail and the ability to work on multiple projects simultaneously."**
    * **Project Alignment:** The meticulous implementation of engineering formulas, the handling of various cable layers, and the inclusion of compliance checks demonstrate a strong attention to detail. The modular design of the `CableDesigner` class allows for the independent definition and analysis of multiple cable designs, reflecting an ability to manage different project aspects.

---

## Installation & Usage (Using Free and Open-Source Tools)

This project is built entirely using **free and open-source software**, ensuring accessibility and transparency.

### Prerequisites

* **Python 3.8+**
* **Git** (for cloning the repository)
* **FreeCAD** (Optional, for 3D model generation, but highly recommended to showcase full functionality. Download from [FreeCAD.org](https://www.freecad.org/))

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/dimitristheodoropoulos/cable_design_project.git]
    cd cable_design_project
    ```
   

2.  **Create and activate a virtual environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate.bat  # On Windows Command Prompt
    # venv\Scripts\Activate.ps1  # On Windows PowerShell
    ```

3.  **Install dependencies:**
    With your virtual environment active, install all required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Run the main script:**
    To execute the cable design calculations and generate reports and plots:
    ```bash
    python 1_Cable_Calculation_Engine.py
    ```
    This will produce `.md` reports, `.json` output files, and `.png` plots in the project directory.

2.  **Explore with Jupyter Notebook (Optional):**
    For an interactive exploration of the code and immediate visualization of results, you can use the Jupyter Notebook:
    ```bash
    jupyter notebook 1_Cable_Calculation_Engine.ipynb
    ```
    (Ensure you run all cells in the notebook to see the outputs and generate updated files.)

3.  **View 3D Models in FreeCAD:**
    Open the `freecad_models/CableCrossSection.FCStd` file directly in FreeCAD to view the generated 3D cross-sections.

## Project Structure
.
├── .gitignore              # Specifies intentionally untracked files to ignore
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── 1_Cable_Calculation_Engine.py  # Main Python script for cable design and analysis
├── 1_Cable_Calculation_Engine.ipynb # Jupyter Notebook for interactive exploration
├── materials.json          # JSON file containing material properties
├── cable_design_output_ex1.json # JSON output for Example 1
├── cable_design_output.json # JSON output for Example 2
├── cable_design_report_example1.md # Markdown report for Example 1
├── cable_design_report_example2.md # Markdown report for Example 2
├── freecad_data.json       # Data/config for FreeCAD integration
├── freecad_models/         # Contains FreeCAD 3D model files
│   └── CableCrossSection.FCStd
├── geometric_dimensions_example_1_standard_cable.png # Plot for geometric dimensions (Ex. 1)
├── geometric_dimensions_example_2_armored_cable.png  # Plot for geometric dimensions (Ex. 2)
└── matplotlib_test_plot.png # Simple test plot for Matplotlib functionality

## Future Enhancements

The modular design of this tool allows for continuous expansion and refinement. Potential future enhancements include:

* Integration of more advanced mechanical analyses (e.g., fatigue, vibration).
* Inclusion of economic optimization modules (e.g., cost-benefit analysis for different designs).
* Graphical User Interface (GUI) development for enhanced user interaction.
* Expansion of compliance checks to cover a wider range of international standards.
* Database integration for larger-scale material and cable management.

## Contact

For any questions, collaboration opportunities, or feedback, please feel free to reach out:

Dimitris Theodoropoulos
dimitris.theodoropoulos@gmail.com