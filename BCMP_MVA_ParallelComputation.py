import numpy as np
from numpy.linalg import solve
import pandas as pd
import time
import sys
from mpi4py import MPI
import itertools
import psutil
import random
import os
import matplotlib.pyplot as plt


class BCMP_MVA_Computation:
    
    def __init__(self, N, R, K, mu, m, K_total, max_x, max_y, tp, pid, rank, size, comm):
        self.N = N
        self.R = R
        self.K = K
        self.K_total = K_total
        self.rank = rank
        self.size = size
        self.comm = comm
        self.pid = pid #Process id
        self.mu = mu #(R×N)
        self.m = m
        self.position = np.zeros((N, 2))
        self.popularity = [[]]
        self.max_distance_x = 500
        self.max_distance_y = 500
        self.maxval = 0
        #Transition Probability generate
        lmd_list = [[]]
        alpha = [[]]
        if rank == 0:
            self.p = tp
            lmd_list = np.zeros((self.R, (np.max(self.K)+1)**self.R), dtype= float)
            self.Pi_list = np.zeros((self.N, np.max(self.m), (np.max(self.K)+1)**self.R), dtype= float)
            alpha = self.getArrival(self.p)
            print('rank = {0}'.format(self.rank))
        self.lmd_list = comm.bcast(lmd_list, root=0)
        self.alpha = comm.bcast(alpha, root=0) 
        self.km = (np.max(self.K)+1)**self.R
        self.output_dir = f'N{N}_R{R}_K{K_total}_U{U}_X{max_x}_Y{max_y}'
        self.cpu_list = []
        self.mem_list = []
        self.roop_time = []
        self.combi_len = []
        self.start = time.time()
        

    def getMVA(self):
        state_list = [] #State number of the previous hierarchy 
        l_value_list =[] #L value for the previous state 
        state_dict = {} #Dict of {l_value_list, state_list} for the previous state
        last_L = []
        for k_index in range(1, self.K_total+1):
            if self.rank == 0:
                process = psutil.Process(pid=self.pid)
                cpu = psutil.cpu_percent(interval=1)
                self.cpu_list.append(cpu)
                self.mem_list.append(process.memory_info().rss/1024**2) #MB
                self.roop_time.append(time.time() - self.start)
            else:
                process = psutil.Process(pid=self.pid)
                self.mem_list.append(process.memory_info().rss/1024**2) #MB

            k_combi_list_div_all = [[]for i in range(self.size)]
            if self.rank == 0:
                k_combi_list = self.getCombiList4(self.K, k_index)
                self.combi_len.append(len(k_combi_list))
                k_combi_list_div = k_combi_list_div_all[0]
                quotient, remainder = divmod(len(k_combi_list), self.size)
                size_index = 0
                for k_combi_list_val in k_combi_list:
                    k_combi_list_div_all[size_index % self.size].append(k_combi_list_val)
                    size_index += 1
                for rank in range(1, self.size):
                    self.comm.send(k_combi_list_div_all[rank], dest=rank, tag=10)
            else :
                k_combi_list_div = self.comm.recv(source=0, tag=10)
                quotient, remainder = None, None
            quotient = self.comm.bcast(quotient, root=0)
            remainder = self.comm.bcast(remainder, root=0)
            
            #Used for passing within parallel loops 
            L = np.zeros((self.N, self.R, len(k_combi_list_div)), dtype= float) #Average number of people in the system 
            T = np.zeros((self.N, self.R, len(k_combi_list_div)), dtype= float) #Average time in the system 
            lmd = np.zeros((self.R, len(k_combi_list_div)), dtype= float) #Throughput for each class 
            Pi = np.zeros((self.N, np.max(self.m), len(k_combi_list_div)*self.R), dtype= float)

            for idx, val in enumerate(k_combi_list_div): # Implemented only in my charge
                # Update T
                k_state_number = int(self.getState(val)) # Convert state of k to decimal
                for n in range(self.N): # P336 (8.43)
                    for r in range(self.R):
                        r1 = np.zeros(self.R) # Compute K-1r
                        r1[r] = 1 # Set only the target class to 1
                        k1v = val - r1 # Vector subtraction

                        # Master process (rank 0) gets values used in recursive computation
                        if self.m[n] > 1:
                            pi_list = [[] for i in range(self.size)] # Values used in recursive computation
                            if self.rank == 0:
                                rank_list = [] # Store ranks that performed computation
                                end = remainder
                                if quotient > idx:
                                    end = self.size # All processes perform computation
                                # Get ranks that performed computation, store in rank_list
                                for rank in range(1, end):
                                    rank_number = self.comm.recv(source=rank, tag=15)
                                    rank_list.append(rank_number)
                                if len(rank_list) > 0: # Register values to pi_list
                                    for rank in rank_list:
                                        if rank > 0:
                                            rank_idx = self.comm.recv(source=rank, tag=16)
                                            for i in range(len(rank_idx[0])):
                                                pi_list[rank].append(self.Pi_list[n][int(rank_idx[0][i])][int(rank_idx[1][i])]) # self.Pi_list[n][j-1][kr_state]
                                    for rank in range(1, end):
                                        self.comm.send(pi_list, dest=rank, tag=19)
                            else:
                                # Send indexes of target values to rank 0
                                if np.min(k1v) >= 0 and np.sum(k1v) > 0:
                                    self.comm.send(self.rank, dest=0, tag=15) # Send rank number if computed
                                    pi_idx = [[] for i in range(2)] # Store state_number and j-1
                                    for j in range(1, self.m[n]):
                                        for _r in range(self.R):
                                            kr = np.zeros(self.R)
                                            kr[_r] = 1
                                            kr_state = int(self.getState_kr(k1v, kr)) # k1v is the previous state of val
                                            if kr_state < 0:
                                                continue
                                            pi_idx[0].append(j-1)
                                            pi_idx[1].append(kr_state)
                                    self.comm.send(pi_idx, dest=0, tag=16)
                                else:
                                    self.comm.send(0, dest=0, tag=15) # Send 0 if not computed
                            # Receive values from rank 0
                            if self.rank > 0:
                                pi_list = self.comm.recv(source=0, tag=19)

                        if np.min(k1v) >= 0:
                            kr_state_number = int(self.getState(k1v)) # Get index position of k-1r
                            # Get previous L and its sum
                            sum_l = 0
                            for i in range(self.R): # Convert k-1r to state
                                l_value = state_dict.get((kr_state_number,n,i)) # Search state_list and return l_value
                                if l_value is not None:
                                    sum_l += l_value 

                            if self.m[n] == 1:
                                T[n, r, idx] = 1 / self.mu[r,n] * (1 + sum_l)
                            if self.m[n] > 1:
                                sum_pi = 0
                                for _j in range(m[n]-2+1):
                                    # k1v is combination, kr_state_number is index
                                    pi = self.getPi(n, _j, k1v, kr_state_number, Pi, idx, r, pi_list)
                                    Pi[n][_j][idx*self.R + r] = pi
                                    if self.rank == 0: # Only rank 0 adds it later
                                        self.Pi_list[n][_j][kr_state_number] = pi
                                    sum_pi += (self.m[n] - _j - 1) * pi
                                T[n, r, idx] = 1 / (self.m[n] * self.mu[r,n]) * (1 + sum_l + sum_pi)

                # Update λ
                for r in range(self.R):
                    sum = 0
                    for n in range(self.N):
                        sum += self.alpha[r][n] * T[n,r,idx]
                    if sum == 0:
                        continue
                    if sum > 0:
                        lmd[r,idx] = val[r] / sum
                        if self.rank == 0: # Only rank 0 adds it later
                            self.lmd_list[r][k_state_number] = lmd[r][idx]

                # Update L
                for n in range(self.N):
                    for r in range(self.R):
                        L[n,r,idx] = lmd[r,idx] * T[n,r,idx] * self.alpha[r][n]

            # Gather and broadcast entire process
            state_list = []
            l_value_list =[]
            state_dict = {}
            n_list = []
            r_list = []
            if self.rank == 0:
                for idx, j in enumerate(k_combi_list_div):
                    k_state = int(self.getState(j))
                    for n in range(self.N): # Update L
                        for r in range(self.R):
                            state_list.append(k_state)
                            l_value_list.append(L[n,r,idx])
                            n_list.append(n)
                            r_list.append(r)
                for i in range(1, self.size):
                    lmd_rank = self.comm.recv(source=i, tag=11)
                    l_rank = self.comm.recv(source=i, tag=12)
                    Pi_rank = self.comm.recv(source=i, tag=13)
                    # Merge lists
                    for idx, j in enumerate(k_combi_list_div_all[i]):
                        k_state = int(self.getState(j)) # Convert state of k to decimal
                        for r in range(self.R): # Update Lambda
                            self.lmd_list[r,k_state] = lmd_rank[r,idx]    
                        for n in range(self.N): # Update L
                            for r in range(self.R):
                                state_list.append(k_state)
                                l_value_list.append(l_rank[n,r,idx])
                                n_list.append(n)
                                r_list.append(r)
                        # Merge Pi_list
                        for r in range(self.R):
                            kr = np.zeros(self.R)
                            kr[r] = 1
                            kr_state = int(self.getState_kr(j, kr))
                            if kr_state < 0:
                                continue
                            else:
                                for n in range(self.N):
                                    for j in range(np.max(self.m)):
                                        self.Pi_list[n,j,kr_state] = Pi_rank[n,j,idx*self.R + r]
            else:
                self.comm.send(lmd, dest=0, tag=11)
                self.comm.send(L, dest=0, tag=12)
                self.comm.send(Pi, dest=0, tag=13)
            self.comm.barrier()
 
            
            
            #broadcast
            self.lmd_list = self.comm.bcast(self.lmd_list, root=0)
            state_list = self.comm.bcast(state_list, root=0)
            l_value_list = self.comm.bcast(l_value_list, root=0)
            if k_index == self.K_total:
                last_L = l_value_list
            n_list = self.comm.bcast(n_list, root=0)
            r_list = self.comm.bcast(r_list, root=0)
            state_dict = dict(zip(zip(state_list, n_list, r_list),l_value_list))

        if self.rank == 0:
            for i in range(1, self.size):
                mem_rank = self.comm.recv(source=i, tag=14)
                self.mem_list = np.add(self.mem_list, mem_rank)
            df_info = pd.DataFrame({ 'combination': self.combi_len, 'memory' : self.mem_list, 'cpu' : self.cpu_list, 'elapse' : self.roop_time})
            df_info.to_csv(f'./{self.output_dir}/computation_info_Core'+str(self.size)+'.csv', index=True)
        else:
            self.comm.send(self.mem_list, dest=0, tag=14)
        self.comm.barrier() 

        return last_L


    def getPi(self, n, j, k, k_state, Pi, idx, r, pi_list):
        if j == 0 and sum(k) == 0:
            return 1
        if j > 0 and sum(k) == 0: 
            return 0
        if j == 0 and sum(k) > 0: 
            sum_emlam = 0
            for _r in range(self.R):
                sum_emlam += self.alpha[_r][n] / self.mu[_r][n] * self.lmd_list[_r][k_state]
            sum_pi = 0
            i = 0
            for _j in range(1, self.m[n]):
                pi8_44, i = self.getPi8_44(n, _j, k, k_state, pi_list, i) 
                Pi[n][_j][idx*self.R + r] = pi8_44
                if self.rank == 0: 
                    self.Pi_list[n][_j][k_state] = pi8_44
                sum_pi += (self.m[n] - _j) * pi8_44
            pi = 1 - 1 / self.m[n] * (sum_emlam + sum_pi)
            if pi < 0:
                pi = 0
            return pi
        if j > 0 and sum(k) > 0:
            return Pi[n][j][idx*self.R + r]

    def getPi8_44(self, n, j, k, k_state, pi_list, i):
        sum_val = 0
        for _r in range(self.R):
            kr = np.zeros(self.R)
            kr[_r] = 1
            kr_state = int(self.getState_kr(k, kr))
            if kr_state < 0:
                continue
            else:
                if self.rank == 0:
                    sum_val += self.alpha[_r][n] / self.mu[_r][n] * self.lmd_list[_r][k_state] * self.Pi_list[n][j-1][kr_state]
                else:   
                    sum_val += self.alpha[_r][n] / self.mu[_r][n] * self.lmd_list[_r][k_state] * pi_list[self.rank][i] #pi_list[self.rank][i] 
                    i += 1
        return 1 / j * (sum_val), i #Pi[n][j][idx*self.R + r]

    
    def getState(self, k): # When k = [k1, k2, ...] is passed as an argument, return its value in base-n (R = len(K))
        k_state = 0
        for i in range(self.R): # Compute the state of k for determining L (e.g., in base-3)
            k_state += k[i]*((np.max(self.K)+1)**int(self.R-1-i))
        return k_state

    def getState_kr(self, k, kr): # Previous state before Pi
        kr_state = 0
        kkr = k - kr
        if min(kkr) < 0:
            return -1
        else:
            for i in range(self.R):
                kr_state += kkr[i]*((np.max(self.K)+1)**int(self.R-1-i))
            return kr_state

    def getArrival(self, p): # Calculate arrival rate for each class from multiclass transition probabilities
        p = np.array(p) # Convert list to numpy array (for convenience)
        alpha = np.zeros((self.R, self.N))
        for r in range(self.R): # Extract each multiclass and compute arrival rate
            alpha[r] = self.getCloseTraffic(p[r * self.N : (r + 1) * self.N, r * self.N : (r + 1) * self.N])
        return alpha

    
    # Calculate node arrival rate α for a closed network
    def getCloseTraffic(self, p):
        e = np.eye(len(p)-1) # Reduce the dimension by one
        pe = p[1:len(p), 1:len(p)].T - e # Reduce the dimension by selecting rows and columns
        lmd = p[0, 1:len(p)] # Use values from row 0, columns 1 onward as the right-hand side
        try:
            slv = solve(pe, lmd * (-1)) # 2021/09/28 Error occurs here if inverse matrix does not exist
        except np.linalg.LinAlgError as err: # 2021/09/29 Handle "Singular Matrix" by adding small value to diagonal elements https://www.kuroshum.com/entry/2018/12/28/python%E3%81%A7singular_matrix%E3%81%8C%E8%B5%B7%E3%81%8D%E3%82%8B%E7%90%86%E7%94%B1
            print('Singular Matrix')
            pe += e * 0.00001 
            slv = solve(pe, lmd * (-1)) 
        alpha = np.insert(slv, 0, 1.0) # Add α1=1
        return alpha    
    
    def getCombiList4(self, K, Pnum): # For parallel computation: compute in parallel by increasing Pnum (2022/1/19)
        # Klist: max number of people at each node; Pnum: total number of people to be distributed
        Klist = [[j for j in range(K[i]+1)] for i in range(len(K))]
        combKlist = list(itertools.product(*Klist))
        combK = [list(cK) for cK in combKlist if sum(cK) == Pnum ]
        return combK
    
    

    


if __name__ == '__main__':
   
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    pid = os.getpid() #Process id

    N = int(sys.argv[1]) #Number of nodes
    R = int(sys.argv[2]) #Number of classes
    K_total = int(sys.argv[3]) 
    U = int(sys.argv[4]) 
    
    if U > 1 and size > 1:
        if rank == 0:
            print("Parallelization with multiple servers is not supported")
        sys.exit()  
    
    max_x = int(sys.argv[5]) if len(sys.argv) > 5 else 500
    max_y = int(sys.argv[6]) if len(sys.argv) > 6 else 500
    
    output_dir = f'N{N}_R{R}_K{K_total}_U{U}_X{max_x}_Y{max_y}'
    p = np.loadtxt(f'{output_dir}/transition_probability.csv', delimiter=',').tolist()
    m = np.loadtxt(f'{output_dir}/m_values.csv', dtype=int)
    mu = np.loadtxt(f'{output_dir}/mu_matrix.csv', delimiter=',')
    K = np.loadtxt(f'{output_dir}/K_values.csv', delimiter=',').astype(int).tolist()
    
    
    bcmp = BCMP_MVA_Computation(N, R, K, mu, m, K_total, max_x, max_y, p, pid, rank, size, comm)
    start = time.time()
    L = bcmp.getMVA()
    elapsed_time = time.time() - start
    print ("calclation_time:{0}".format(elapsed_time) + "[sec]")
    sum = 0
    if rank == 0:
        Lr = np.zeros((N, R))
        for n in range(N):
            for r in range(R):
                sum += L[(n*R+r)]
                Lr[n, r] = L[(n*R+r)]
        print(sum)
        print(f'L =\n{Lr}')
        np.savetxt(f'{output_dir}/MVA_L_matrix_Core'+str(size)+'.csv', Lr, delimiter=',')
       
        

    #mpiexec -n 8 python BCMP_MVA_ParallelComputation.py 10 2 50 1
    #mpiexec -n 1 python BCMP_MVA_ParallelComputation.py 10 2 50 3
