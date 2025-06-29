BCMP_Graph - Visualization of BCMP Queueing Network Results
============================================================

Overview:
---------
This script generates visualizations from BCMP simulation or MVA outputs.
It includes:
- A heatmap of spatial customer density
- A bar chart of class-wise load per node

Useful for understanding class distribution and service efficiency.

Requirements:
-------------
- Python 3.x
- numpy, pandas, matplotlib

Usage:
------
python BCMP_Graph.py <output_directory>

Examples:
---------
python BCMP_Graph.py N10_R2_K50_U3_X500_Y500
python BCMP_Graph.py Optimization_N10_R2_K50_U3_X500_Y500_algorithmSimulation_Size01_20250627194740

Input Files (in <output_directory>):
------------------------------------
- transition_probability.csv
- m_values.csv
- node_info.csv (must include `position_x`, `position_y`)
- One of: MVA_L_matrix*.csv, ga_L_std.csv, avg_Lc_*.csv

Output:
-------
- heatmap_<target>.png : Spatial distribution of customers
- L_<target>.png : Class-wise load per node

Notes:
------
- The script expects the directory name to contain `X###_Y###` to set grid size
- Labels are styled based on popularity and max window usage

License:
--------
MIT License or as specified

Author:
-------
(Your name or institution here)