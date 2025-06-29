# BCMP Network Analysis and Optimization - README

This README provides a unified guide to the following Python modules for simulating, analyzing, and optimizing BCMP queueing networks:

1. `BCMP_makeTransitionProbability.py` – Transition Probability Generation  
2. `BCMP_MVA.py` – Mean Value Analysis (MVA)
3. `BCMP_MVA_ParallelComputation.py` – Mean Value Analysis (MVA)
4. `BCMP_Simulation.py` – Network Simulation  
5. `BCMP_Optimization.py` – Server Allocation Optimization  
6. `BCMP_Graph.py` - Output structure and visualizations (Graph section)  

---

## 1. BCMP_makeTransitionProbability.py

### Overview
This script generates node positions, popularity levels, and a transition probability matrix based on a gravity model using inter-node distances and popularity.

### Usage
```bash
python BCMP_makeTransitionProbability.py N R K_total U [max_distance_x] [max_distance_y]
```

### Example
```bash
python BCMP_makeTransitionProbability.py 10 2 50
```

### Output
A directory will be created in the format:
```bash
./N{N}_R{R}_K{K_total}_U{U}_X{max_distance_x}_Y{max_distance_y}/
```

Containing:  
`transition_probability.csv`  
`distance.csv`  
`distance_matrix.csv`  
`popularity.csv`  
`node_info.csv`  
`position.csv`  


## 2. BCMP_MVA.py 

### Overview
Performs Mean Value Analysis (MVA) for a closed BCMP network with multiple customer classes. Calculates average number of customers in each node (L).

### Usage
```bash
python BCMP_MVA.py N R K_total U [max_distance_x] [max_distance_y]
```

### Example
```bash
python BCMP_MVA.py 10 2 50 3
```

### Output
CSV file:  
`MVA_N{N}_R{R}_K{K_total}_U{U}_X{max_distance_x}_Y{max_distance_y}/L_matrix.csv`  



## 3. BCMP_MVA_ParallelComputation.py

### Overview
This script performs parallelized Mean Value Analysis (MVA) for a closed BCMP queueing network with multiple customer classes.  
It distributes the computation of class and customer configurations across multiple MPI processes, enabling faster performance for large-scale networks.  
⚠️ **Limitations**  
Currently, this parallel implementation **does not support configurations where both the number of MPI processes ≥ 2 and the number of service servers per node ≥ 2**.  
If you require multiple servers per node, please run the non-parallel version (`BCMP_MVA.py`) or restrict the window count to 1 when using parallel execution.


### Usage
Run using `mpiexec` with the desired number of processes:
```bash
mpiexec -n <num_processes> python BCMP_MVA_ParallelComputation.py N R K_total [max_distance_x] [max_distance_y]
```

### Example
```bash
mpiexec -n 8 python BCMP_MVA.py 10 2 50 1
```

### Output
CSV file:  
`MVA_N{N}_R{R}_K{K_total}_U{U}_X{max_distance_x}_Y{max_distance_y}/L_matrix_Core{num_processes}.csv`  



## 4. BCMP_Simulation.py

### Overview
Performs discrete-event simulation of BCMP queueing networks with MPI-based parallelism. Calculates RMSE versus theoretical MVA results, and logs performance metrics.

### Required Input Files
- `transition_probability.csv`, `m_values.csv`, `mu_matrix.csv`, `K_values.csv`
- `MVA_L_matrix_*.csv` for theoretical comparison

### Usage
Run using `mpiexec` with the desired number of processes:
```bash
mpiexec -n <num_processes> python BCMP_Simulation.py N R K_total U max_x max_y sim_time
```

### Example
```bash
mpiexec -n 8 python BCMP_MVA.py 10 2 50 1 3 500 500 10000
```

### Output
Automatically generated directory with:
- `L_Rank_*.csv`, `Q_Rank_*.csv`, `Lc_Rank_*.csv`, `Qc_Rank_*.csv`
- `RMSE_Rank_*.csv`, `RMSE.png`, `avg_RMSE_*.png`
- `length_Rank_*.png`, `model_info.csv`
- Performance logs: `Time_Rank_*.csv`, `memory_Rank_*.csv`



## 5. BCMP_Optimization.py

### Overview
Uses Genetic Algorithm (GA) to optimize server (window) allocation in a BCMP network. Objective: minimize the sum of the standard deviation of customer load (L) and deployment cost.

### Required Input Files
- `transition_probability.csv`, `mu_matrix.csv`, `K_values.csv`
- Uses MVA or Simulation for evaluation

### Usage
Run using `mpiexec` with the desired number of processes:
```bash
mpiexec -n <num_processes> python BCMP_Optimization.py N R K_total U max_x max_y npop ngen crosspb mutpb P algorithm [sim_time]
```

### Example
Using MVA:
```bash
mpiexec -n 8 python BCMP_Optimization.py 10 2 50 1 500 500 8 20 0.5 0.2 1 1
```
Using Simulation:
```bash
mpiexec -n 8 python BCMP_Optimization.py 10 2 50 1 500 500 8 20 0.5 0.2 1 2 10000
```

### Output
Generated directory includes:
- `ga_L_std.csv`, `ga_Node_std.csv`, `ga_P_std.csv`
- `ga_Object_std.csv`, `ga_transition_std.png`
- `run_info.txt`


## 6. BCMP_Graph.py

### Usage
```bash
python BCMP_Graph.py <folderpath>
```

### Example
```bash
python BCMP_Graph.py N10_R2_K50_U1_X500_Y500 
```


### Graphs
- `heatmap_*.png`: Heatmap showing node positions and load distribution
- `L_*.png`: Bar chart of average number of customers per node and class


## License
MIT License

## Corresponding contributor
Shinya Mizuno
s.mzn.eng@gmail.com












