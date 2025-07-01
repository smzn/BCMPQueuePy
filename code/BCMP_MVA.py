import numpy as np
from numpy.linalg import solve
import pandas as pd
import time
import sys
import os


class BCMP_MVA:
    def __init__(self, N, R, K, mu, p, m):
        self.N = N  # Number of nodes
        self.R = R  # Number of classes
        self.K = K  # Customers per class
        self.mu = mu  # Service rate (R x N)
        self.p = p  # Transition probability matrix
        self.alpha = self.get_arrival_rates(self.p)  # Arrival rate matrix
        self.m = m  # Number of servers per node

        self.combi_list = self.generate_combinations([], self.K, self.R, 0, [])
        self.km = (np.max(self.K) + 1) ** self.R  # Number of possible states

        self.L = np.zeros((self.N, self.R, self.km), dtype=float)  # Mean number in system
        self.T = np.zeros((self.N, self.R, self.km), dtype=float)  # Mean response time
        self.lmd = np.zeros((self.R, self.km), dtype=float)  # Throughput per class
        self.Pi = np.zeros((self.N, np.max(self.m), self.km), dtype=float)

    def get_MVA(self):
        for idx, val in enumerate(self.combi_list):
            if idx == 0:
                continue

            k_state = self.get_state_index(val)

            # Update T
            for n in range(self.N):
                for r in range(self.R):
                    r1 = np.zeros(self.R)
                    r1[r] = 1
                    k1v = val - r1

                    if np.min(k1v) < 0:
                        continue

                    kn = int(self.get_state_index(k1v))
                    sum_l = sum(self.L[n, i, kn] for i in range(self.R))

                    if self.m[n] == 1:
                        self.T[n, r, k_state] = (1 / self.mu[r, n]) * (1 + sum_l)
                    elif self.m[n] > 1:
                        sum_pi = 0
                        for j in range(self.m[n] - 1):
                            pi_val = self.get_Pi(n, j, val, r1)
                            self.Pi[n, j, kn] = pi_val
                            sum_pi += (self.m[n] - j - 1) * pi_val
                        self.T[n, r, k_state] = (1 / (self.m[n] * self.mu[r, n])) * (1 + sum_l + sum_pi)
                    else:
                        self.T[n, r, k_state] = 1 / self.mu[r, n]

            # Update lambda
            for r in range(self.R):
                total = sum(self.alpha[r, n] * self.T[n, r, k_state] for n in range(self.N))
                if total > 0:
                    self.lmd[r, k_state] = val[r] / total

            # Update L
            for n in range(self.N):
                for r in range(self.R):
                    self.L[n, r, k_state] = self.lmd[r, k_state] * self.T[n, r, k_state] * self.alpha[r, n]

        final_state = self.get_state_index(self.combi_list[-1])
        return self.L[:, :, final_state]

    def get_Pi(self, n, j, k, kr):
        kkr = k - kr
        state_number = int(self.get_state_index(kkr))

        if np.min(kkr) < 0:
            return 0
        if j == 0 and sum(kkr) == 0:
            return 1
        if j > 0 and sum(kkr) == 0:
            return 0
        if j == 0:
            sum_emlam = sum(self.alpha[r, n] / self.mu[r, n] * self.lmd[r, state_number] for r in range(self.R))
            sum_pi = 0
            for j2 in range(1, self.m[n]):
                pi_val = self.get_Pi_8_44(n, j2, kkr, state_number)
                self.Pi[n, j2, state_number] = pi_val
                sum_pi += (self.m[n] - j2) * pi_val
            pi = 1 - (1 / self.m[n]) * (sum_emlam + sum_pi)
            return max(pi, 0)
        else:
            return self.Pi[n, j, state_number]

    def get_Pi_8_44(self, n, j, k, state_number):
        result = 0
        for r in range(self.R):
            kr = np.zeros(self.R)
            kr[r] = 1
            kr_state_number = int(self.get_state_index(k - kr))
            if kr_state_number >= 0:
                result += self.alpha[r, n] / self.mu[r, n] * self.lmd[r, state_number] * self.Pi[n, j - 1, kr_state_number]
        return result / j

    def get_state_index(self, k):
        return int(sum(k[i] * ((np.max(self.K) + 1) ** (self.R - 1 - i)) for i in range(self.R)))

    def get_arrival_rates(self, p):
        p = np.array(p)
        alpha = np.zeros((self.R, self.N))
        for r in range(self.R):
            sub_p = p[r * self.N:(r + 1) * self.N, r * self.N:(r + 1) * self.N]
            alpha[r] = self.solve_closed_traffic(sub_p)
        return alpha

    def solve_closed_traffic(self, p):
        I = np.eye(len(p) - 1)
        pe = p[1:, 1:].T - I
        rhs = -p[0, 1:]

        try:
            sol = solve(pe, rhs)
        except np.linalg.LinAlgError:
            print('Singular Matrix')
            pe += I * 1e-5
            sol = solve(pe, rhs)

        return np.insert(sol, 0, 1.0)

    def generate_combinations(self, combi, K, R, idx, result):
        if len(combi) == R:
            result.append(np.array(combi.copy()))
            return result
        for v in range(K[idx] + 1):
            combi.append(v)
            self.generate_combinations(combi, K, R, idx + 1, result)
            combi.pop()
        return result


if __name__ == '__main__':
    N = int(sys.argv[1])
    R = int(sys.argv[2])
    K_total = int(sys.argv[3])
    U = int(sys.argv[4]) 

    max_x = int(sys.argv[5]) if len(sys.argv) > 5 else 500
    max_y = int(sys.argv[6]) if len(sys.argv) > 6 else 500
    
    output_dir = f'../results/N{N}_R{R}_K{K_total}_U{U}_X{max_x}_Y{max_y}'
    p = np.loadtxt(f'{output_dir}/transition_probability.csv', delimiter=',').tolist()
    m = np.loadtxt(f'{output_dir}/m_values.csv', dtype=int)
    mu = np.loadtxt(f'{output_dir}/mu_matrix.csv', delimiter=',')
    K = np.loadtxt(f'{output_dir}/K_values.csv', delimiter=',').astype(int).tolist()
    

    bcmp = BCMP_MVA(N, R, K, mu, p, m)

    start = time.time()
    L = bcmp.get_MVA()
    elapsed = time.time() - start

    print(f"Calculation time: {elapsed:.4f} seconds")
    print(f'L =\n{L}')

    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(f'{output_dir}/MVA_L_matrix.csv', L, delimiter=',')


#python BCMP_MVA.py 10 2 50 1
#python BCMP_MVA.py 10 2 50 3