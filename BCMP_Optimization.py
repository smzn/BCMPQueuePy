import math
import matplotlib.pyplot as plt
import time
import sys
import numpy as np
from numpy.random import randint, rand
import random
import os
from mpi4py import MPI


class BCMP_GA_Class:

    def __init__(self, N, R, npop, ngen, U, crosspb, mutpb, rank, size, comm, P, sim_time, tp, algorithm,mu, K, dirname):
        self.dirname = dirname
        self.N = N
        self.R = R
        self.K = K
        self.mu = mu
        self.P = P
        self.sim_time = sim_time #simulation time
        self.npop = npop #Population size
        self.ngen = ngen #Genelation size
        self.U = U #Maxmum number of windows
        self.crosspb = crosspb #crossover rate
        self.mutpb = mutpb #mutation rate
        self.rank = rank
        self.size = size
        self.comm = comm
        self.scores = [0 for i in range(self.npop)] # Score for each gene#各遺伝子のスコア
        self.bestfit_seriese = [] # List to store best gene fitness over generations#最適遺伝子適合度を入れたリスト
        self.mean_bestfit_seriese = [] # List to store average fitness of all genes#遺伝子全体平均の適合度
        self.algorithm = algorithm # 1 for MVA, 2 for Simulation
        self.switch = 2 # Used when algorithm is 2 (Simulation)
        # Broadcast initial genes #初期遺伝子をブロードキャスト
        prate = 0.2 # Popularity rate #人気度の割合
        dim = 2 # Dimensionality of distance between bases #拠点間距離の次元数
        if self.rank == 0:
            self.pool = [[self.getRandInt1() for i in range(self.N)] for j in range(self.npop)]  # Initialize genes / 遺伝子を初期化
            self.p = tp
            if self.algorithm == 2:
                # Initial setup for simulation / シミュレーションの初期設定
                self.event = [[] for i in range(self.N)]  # Events (arrival, departure) at each base / 各拠点で発生したイベント(arrival, departure)を格納
                self.eventclass = [[] for i in range(self.N)]  # Customer class at each event / 各拠点でイベント発生時の客クラス番号
                self.eventqueue = [[] for i in range(self.N)]  # Queue length at each event / 各拠点でイベント発生時のqueueの長さ
                self.eventtime = [[] for i in range(self.N)]  # Time of each event / 各拠点でイベントが発生した時の時刻
                self.queue = np.zeros(self.N)  # Total queue length including service / 各拠点のサービス中を含むqueueの長さ(クラスをまとめたもの)
                self.queueclass = np.zeros((self.N, self.R))  # Queue length per class / 各拠点のサービス中を含むqueueの長さ(クラス別)
                self.classorder = [[] for i in range(self.N)]  # Order of customer classes in queue / 拠点に並んでいる順のクラス番号
                self.window = np.full((self.N, self.U), self.R)  # Service windows with customer class / サービス中の客クラス(serviceに対応)(self.Rは空状態)
                self.service = np.zeros((self.N, self.U))  # Remaining service time / サービス中の客の残りサービス時間

                # Distribute initial customers (initial node is base 0) / 開始時の客の分配 (開始時のノードは拠点番号0)
                elapse = 0
                initial_node = 0  # Used only in Step 1 / Step1でのみ使用

                for i in range(self.R):
                    for _ in range(self.K[i]):
                        initial_node = random.randrange(self.N)  # Randomly choose base for each customer / 最初はランダムにいる拠点を決定
                        self.event[initial_node].append("arrival")  # Register event / イベントを登録
                        self.eventclass[initial_node].append(i)  # Customer class number / 到着客のクラス番号
                        self.eventqueue[initial_node].append(self.queue[initial_node])  # Queue length at arrival / イベントが発生した時のqueueの長さ(到着客は含まない)
                        self.eventtime[initial_node].append(elapse)  # Event time (0 for start) / (移動時間0)
                        self.queue[initial_node] += 1  # Increment total queue at node / 最初はノード0にn人いるとする
                        self.queueclass[initial_node][i] += 1  # Increment class-specific count / 拠点0にクラス別人数を追加
                        self.classorder[initial_node].append(i)  # Add class to queue order / 拠点0にクラス番号を追加

                        # Assign customer to available service window / 空いている窓口に客クラスとサービス時間を登録
                        if self.queue[initial_node] <= self.U:
                            self.window[initial_node][int(self.queue[initial_node] - 1)] = i  # Set class in window / クラス番号
                            self.service[initial_node][int(self.queue[initial_node] - 1)] = self.getExponential(self.mu[i][initial_node])  # Set service time / 窓口客のサービス時間設定

        else:
            self.pool = [[]]
            self.p = [[]]
            if self.algorithm == 2:
                self.event = [[]]
                self.eventclass = [[]]
                self.eventqueue = [[]]
                self.eventtime = [[]]
                self.queue = [[]]
                self.queueclass = [[]]
                self.classorder = [[]]
                self.window = [[]]
                self.service = [[]]
        self.pool = self.comm.bcast(self.pool, root=0)
        self.p = self.comm.bcast(self.p, root=0)
        if self.algorithm == 2:
            self.event = self.comm.bcast(self.event, root=0)
            self.eventclass = self.comm.bcast(self.eventclass, root=0)
            self.eventqueue = self.comm.bcast(self.eventqueue, root=0)
            self.eventtime = self.comm.bcast(self.eventtime, root=0)
            self.queue = self.comm.bcast(self.queue, root=0)
            self.queueclass = self.comm.bcast(self.queueclass, root=0)
            self.classorder = self.comm.bcast(self.classorder, root=0)
            self.window = self.comm.bcast(self.window, root=0)
            self.service = self.comm.bcast(self.service, root=0)
        print(self.pool)

    #https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
    # genetic algorithm
    def genetic_algorithm(self):
        best, best_eval = self.pool[0], 10**5
        for gen in range(self.ngen):
            self.scores = [0 for i in range(self.npop)]
            print('generation : {0}'.format(gen))
            for pop_index in range(self.rank, self.npop, self.size):
                start = time.time()
                self.scores[pop_index] = self.getOptimizeBCMP(self.pool[pop_index])
                elapsed_time = time.time() - start
                print ("rank = {1}, pop_id = {2}, calclation_time:{0}".format(elapsed_time, self.rank, pop_index) + "[sec]")
                print('pop_index = {0}, scores = {1}, rank = {2}'.format(pop_index, self.scores[pop_index], self.rank))
            
            #Aggregate data
            if self.rank == 0:
                for i in range(1, self.size):
                    scores = self.comm.recv(source=i, tag=11)
                    for j in range(len(self.scores)):
                        self.scores[j] += scores[j]
            else:
                self.comm.send(self.scores, dest=0, tag=11)
            self.comm.barrier()
            if self.rank == 0:
                print('Gen{1} score : {0}'.format(self.scores, gen))
            
            #check for new best solution
            if self.rank == 0:
                for i in range(self.npop):
                    if self.scores[i] < best_eval: #Find the minimum value
                        best, best_eval = self.pool[i], self.scores[i]
                        print("Generation : {0}, new best {1} = {2}".format(gen, self.pool[i], self.scores[i]))
                        print('Number of nodes : {0}'.format(sum(self.pool[i])))
                #select parents
                selected = [self.selection() for c in range(self.npop)] 
                #create the next generation
                children = list()
                for i in range(0, self.npop, 2):
                    #get selected parents in pairs
                    p1, p2 = selected[i], selected[i+1] #Error if population size is odd
                    #crossover and mutation
                    for c in self.crossover(p1, p2):
                        #mutation
                        self.mutation(c)
                        #store for next generation
                        children.append(c)
                #replace population
                self.pool = children
                # Save objective function values for each generation / 世代毎の目的関数値を保存
                self.bestfit_seriese.append(best_eval)
                self.mean_bestfit_seriese.append(sum(self.scores)/len(self.scores))
            #broadcast
            self.pool = self.comm.bcast(self.pool, root=0)
            best = self.comm.bcast(best, root=0)
            best_eval = self.comm.bcast(best_eval, root=0)
            
        if self.rank == 0:
            self.getGraph()
            #Final Result
            self.getFinalResult(best)
        return [best, best_eval]
          
    def getFinalResult(self, individual):
        if self.algorithm == 1: #MVA
            import BCMP_MVA as mdl
            bcmp_mva = mdl.BCMP_MVA(self.N, self.R, self.K, self.mu, self.p, individual)
            theoretical = bcmp_mva.get_MVA()
            L_class = np.array(theoretical) #list to numpy
        if self.algorithm == 2: #Simulation
            #Modify arrays (service, window)
            service = np.zeros((self.N, self.U))
            window = np.full((self.N, self.U), self.R)
            for i in range(len(individual)): 
                for j in range(individual[i]):
                    service[i][j] = self.service[i][j]
                    window[i][j] = self.window[i][j]
            #simulate
            import BCMP_Simulation as mdl
            bcmp_simulation = mdl.BCMP_Simulation(self.N, self.R, self.K, self.U, self.mu, individual, self.p, self.sim_time, self.switch, self.event, self.eventclass, self.eventqueue, self.eventtime, self.queue, self.queueclass, self.classorder, window, service)
            simulation = bcmp_simulation.getSimulation()
            L_class = np.array(simulation) #list to numpy
        np.savetxt(f'{self.dirname}/ga_L_std.csv', L_class, delimiter=',')
        np.savetxt(f'{self.dirname}/ga_Node_std.csv', individual, delimiter=',') 
        np.savetxt(f'{self.dirname}/ga_P_std.csv', self.p, delimiter=',')
        np.savetxt(f'{self.dirname}/ga_Object_std.csv', np.array(self.bestfit_seriese), delimiter=',')
        print('Final Result')
        print('L = {0}'.format(L_class))
        print('sum = {0}'.format(np.sum(L_class)))
        print('Node = {0}'.format(individual))
           
    def getOptimizeBCMP(self, individual):
        if self.algorithm == 1: #MVA
            import BCMP_MVA as mdl
            bcmp_mva = mdl.BCMP_MVA(self.N, self.R, self.K, self.mu, self.p, individual)
            theoretical = bcmp_mva.get_MVA()
            L_class = np.array(theoretical) #list to numpy
        if self.algorithm == 2: #Simulation
            #Modify arrays (service, window)
            service = np.zeros((self.N, self.U))
            window = np.full((self.N, self.U), self.R)
            for i in range(len(individual)): 
                for j in range(individual[i]):
                    service[i][j] = self.service[i][j]
                    window[i][j] = self.window[i][j]
            #simulate
            import BCMP_Simulation as mdl
            bcmp_simulation = mdl.BCMP_Simulation(self.N, self.R, self.K, self.U, self.mu, individual, self.p, self.sim_time, self.switch, self.event, self.eventclass, self.eventqueue, self.eventtime, self.queue, self.queueclass, self.classorder, window, service)
            simulation = bcmp_simulation.getSimulation()
            L_class = np.array(simulation) #list to numpy
        L = []
        for i in range(len(L_class)):
            total = 0
            for j in range(len(L_class[i])):
                total += L_class[i,j]
            L.append(total)
        return self.getObjective(L, individual)

    #Objective Function
    def getObjective(self, l, individual):
        l = np.array(l)
        l1 = l.reshape(1,-1)
        val = np.std(l1)
        total = 0
        for i in range(self.N):
            total += self.P * (individual[i] - 1)
        return val + total

    # tournament selection
    def selection(self, k=3):
        # first random selection
        selection_ix = randint(self.npop)
        for ix in randint(0, self.npop, k-1):
            # check if better (e.g. perform a tournament)
            if self.scores[ix] < self.scores[selection_ix]:
                selection_ix = ix
        return self.pool[selection_ix]
     
    # crossover two parents to create two children
    def crossover(self, p1, p2):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if rand() < self.crosspb:
            # select crossover point that is not on the end of the string
            pt = randint(1, self.N-2)
            # perform crossover
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]
     
    # mutation operator
    def mutation(self, bitstring):
        for i in range(len(bitstring)):
            # check for a mutation
            if rand() < self.mutpb:
                # flip the bit
                bit = randint(1, self.U+1)
                if bitstring[i] == bit:
                    if self.U == 1:
                        # If there is only one possible value to mutate to, just set it (or do nothing)
                        # 変更できる候補が1つしかない場合はそのまま変える（または何もしない）
                        # When self.U == 1, the only valid value is 1
                        # self.U == 1 なので範囲は 1 のみ2025/06/27
                        bitstring[i] = 1  
                    elif (bitstring[i]-1) / (self.U-1) > rand():
                        bitstring[i] = randint(1, bitstring[i])
                    else:
                        bitstring[i] = randint(bitstring[i]+1, self.U+1)
                else:
                    bitstring[i] = bit

 
    def getGraph(self):
        #Draw a graph
        x_axis = [i for i in range(self.ngen)]
        fig = plt.figure()
        plt.plot(x_axis, self.bestfit_seriese, label='elite')
        plt.plot(x_axis, self.mean_bestfit_seriese, label='mean')
        plt.title('Transition of GA Value')
        plt.xlabel('Generation')
        plt.ylabel('Value of GA')
        plt.grid()
        plt.legend()
        fig.savefig(f'{self.dirname}/ga_transition_std.png')
        
    def getRandInt1(self): # Reflect the ease of returning 1 with the minimum number of used nodes / 1を返すときに最低利用Numbe
        return randint(1, self.U+1)
        
    # Generate random values without duplicates / 重複なしランダム生成
    #https://magazine.techacademy.jp/magazine/21160
    def rand_ints_nodup(self, a, b, k):
        ns = []
        while len(ns) < k:
            n = random.randint(a, b)
            if not n in ns:
                ns.append(n)
        return ns
      

    def getExponential(self, param): #Set service time
        return - math.log(1 - random.random()) / param


        
if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = int(sys.argv[1])#Number of nodes in the network
    R = int(sys.argv[2])#Number of classes
    K_total = int(sys.argv[3])#Total number of people in the network
    U = int(sys.argv[4]) #Maximum number of windows

    max_x = int(sys.argv[5]) 
    max_y = int(sys.argv[6])
    
    npop = int(sys.argv[7]) #Population size
    ngen = int(sys.argv[8]) #Generation size

    output_dir = f'N{N}_R{R}_K{K_total}_U{U}_X{max_x}_Y{max_y}'
    p = np.loadtxt(f'{output_dir}/transition_probability.csv', delimiter=',').tolist()#transition probability matrix
    mu = np.loadtxt(f'{output_dir}/mu_matrix.csv', delimiter=',')
    K = np.loadtxt(f'{output_dir}/K_values.csv', delimiter=',').astype(int).tolist()
    
    crosspb = float(sys.argv[9]) #Crossover rate
    mutpb = float(sys.argv[10]) #Mutation rate
    P = float(sys.argv[11]) #Cost weight
    
    algorithm = int(sys.argv[12]) #1(MVA), 2(Simulaion)
    sim_time = int(sys.argv[13]) if len(sys.argv) > 13 else 0 #Simulation time
    
    
    if rank == 0:
        dirname = time.strftime(f"Optimization_{output_dir}_%Y%m%d%H%M%S")
        os.makedirs(f'./{dirname}', exist_ok=True)
        
        # ログファイル作成
        log_path = os.path.join(dirname, "run_info.txt")
        with open(log_path, 'w') as f:
            f.write(f"Execution Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"N (Number of nodes): {N}\n")
            f.write(f"R (Number of classes): {R}\n")
            f.write(f"K_total (Total people): {K_total}\n")
            f.write(f"U (Max windows): {U}\n")
            f.write(f"Population size (npop): {npop}\n")
            f.write(f"Generation size (ngen): {ngen}\n")
            f.write(f"Crossover probability: {crosspb}\n")
            f.write(f"Mutation probability: {mutpb}\n")
            f.write(f"Algorithm: {algorithm} ({'MVA' if algorithm == 1 else 'Simulation'})\n")
            if algorithm == 2:
                f.write(f"Simulation time: {sim_time}\n")
            f.write(f"P (Cost weight): {P}\n")
            f.write(f"MPI  size: {size}\n")
            f.write(f"Transition Probability file: {output_dir}/transition_probability.csv\n")
            f.write(f"mu Matrix file: {output_dir}/mu_matrix.csv\n")
            f.write(f"K Values file: {output_dir}/K_values.csv\n")
    else:
        dirname = None  # rank != 0 は最初は None

    # Broadcast dirname from rank 0 to all ranks
    dirname = comm.bcast(dirname, root=0)
                
                
    start = time.time()
    bcmp = BCMP_GA_Class(N, R, npop, ngen, U, crosspb, mutpb, rank, size, comm, P, sim_time, p, algorithm,mu, K, dirname)
    best, score = bcmp.genetic_algorithm()
    if rank == 0:
        print('Done!')
        print('f(%s) = %f' % (best, score))
        total = 0
        for i in range(N):
            total += P * (best[i] - 1)
        print('std = {0}   cost = {1}'.format(score - total, total))
        elapsed_time = time.time() - start
        print ("calclation_time:{0}".format(elapsed_time) + "[sec]")
    
    #mpiexec -n 8 python BCMP_Optimization.py 10 2 50 1 500 500 8 20 0.5 0.2 1 1
    #mpiexec -n 8 python BCMP_Optimization.py 10 2 50 1 500 500 8 20 0.5 0.2 1 2 10000
