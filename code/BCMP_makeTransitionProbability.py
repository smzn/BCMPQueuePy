import sys
import random
import numpy as np
import pandas as pd
import os


class TransitionProbabilityCalculator:
    def __init__(self, N, R, K_total, U, max_distance_x, max_distance_y):
        self.N = N
        self.R = R
        self.K_total = K_total
        self.U = U
        self.max_distance_x = max_distance_x
        self.max_distance_y = max_distance_y
        self.position = np.zeros((N, 2))  # Node positions (x, y)
        self.popularity = None
        
    def get_gravity(self, distance_matrix):
        """Create a transition probability matrix using the gravity model."""
        C = 0.1475
        alpha = 1.0
        beta = 1.0
        eta = 0.5
        class_number = self.R
        N = self.N

        tp = np.zeros((N * class_number, N * class_number))
        for r in range(class_number):
            for i in range(N * r, N * (r + 1)):
                for j in range(N * r, N * (r + 1)):
                    d = distance_matrix[i % N][j % N]
                    if d > 0:
                        pi = self.popularity[i % N][r]
                        pj = self.popularity[j % N][r]
                        tp[i][j] = C * (pi ** alpha) * (pj ** beta) / (d ** eta)

        # Normalize each row
        row_sum = np.sum(tp, axis=1)
        for i in range(len(tp)):
            if row_sum[i] > 0:
                tp[i] /= row_sum[i]

        return tp

    def get_transition_probability(self):
        """Main function to create node positions, distances, popularity, and transition matrix."""
        # (1) Generate node positions
        x_positions = [random.randint(0, self.max_distance_x) for _ in range(self.N)]
        y_positions = [random.randint(0, self.max_distance_y) for _ in range(self.N)]

        self.position[:, 0] = x_positions
        self.position[:, 1] = y_positions

        # (2) Create distance matrix and distance dataframe
        distance_matrix = np.zeros((self.N, self.N))
        from_id, to_id, distance = [], [], []

        for i in range(self.N):
            for j in range(i + 1, self.N):
                dist = np.hypot(x_positions[i] - x_positions[j], y_positions[i] - y_positions[j])
                from_id.append(i)
                to_id.append(j)
                distance.append(dist)
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist  # Symmetric

        df_distance = pd.DataFrame({
            'from_id': from_id,
            'to_id': to_id,
            'distance': distance
        })

        # (3) Assign random popularity values
        self.popularity = np.abs(np.random.normal(10, 2, (self.N, self.R)))

        # (4) Compute transition probabilities
        tp = self.get_gravity(distance_matrix)

        # (5) Create node info dataframe
        df_node = pd.DataFrame({
            'node_number': range(self.N),
            'position_x': x_positions,
            'position_y': y_positions
        }).set_index('node_number')

        for r in range(self.R):
            df_node[f'popularity_{r}'] = self.popularity[:, r]
            
        # (6) Assign m values (number of servers per node) based on popularity
        total_popularity = np.sum(self.popularity, axis=1)  # shape: (N,)
        normalized = total_popularity / np.max(total_popularity)  # range: 0–1
        m_values = np.clip(np.round(normalized * self.U), 1, self.U).astype(int)  # range: 1–U
        
        # (8) Generate mu matrix: service rate for each class and node
        mu_matrix = np.abs(np.random.normal(loc=1.0, scale=0.1, size=(self.R, self.N)))
        
        # (9) Generate K: number of customers per class
        K = np.random.multinomial( self.K_total, [1.0 / self.R] * self.R)
        
        # (7) Save data
        dirname = f'../results/N{self.N}_R{self.R}_K{self.K_total}_U{self.U}_X{self.max_distance_x}_Y{self.max_distance_y}'
        os.makedirs(f'./{dirname}', exist_ok=True)

        df_node.to_csv(f'./{dirname}/node_info.csv')
        df_distance.to_csv(f'./{dirname}/distance.csv', index=False)
        np.savetxt(f'./{dirname}/distance_matrix.csv', distance_matrix, delimiter=',')
        np.savetxt(f'./{dirname}/transition_probability.csv', tp, delimiter=',')
        np.savetxt(f'./{dirname}/popularity.csv', self.popularity, delimiter=',')
        np.savetxt(f'./{dirname}/position.csv', self.position, delimiter=',')
        np.savetxt(f'./{dirname}/m_values.csv', m_values, fmt='%d', delimiter=',')
        np.savetxt(f'./{dirname}/mu_matrix.csv', mu_matrix, delimiter=',')
        np.savetxt(f'./{dirname}/K_values.csv', K, fmt='%d', delimiter=',')

        return tp


if __name__ == '__main__':
    # Read input arguments: number of nodes, classes, and total users
    N = int(sys.argv[1])
    R = int(sys.argv[2])
    K_total = int(sys.argv[3]) 
    U = int(sys.argv[4])
    
    # Optional arguments: max_distance_x and max_distance_y
    max_distance_x = int(sys.argv[5]) if len(sys.argv) > 5 else 500
    max_distance_y = int(sys.argv[6]) if len(sys.argv) > 6 else 500

    # Create calculator and compute transition probabilities
    calc = TransitionProbabilityCalculator(N, R, K_total, U, max_distance_x, max_distance_y)
    tp_matrix = calc.get_transition_probability()


# python BCMP_makeTransitionProbability.py 10 2 50 1 500 500 
# python BCMP_makeTransitionProbability.py 10 2 50 3 500 500
