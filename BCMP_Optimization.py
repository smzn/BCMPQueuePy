import math
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import numpy as np
from numpy.random import randint
from numpy.random import rand
import random
import os
from mpi4py import MPI


class BCMP_GA_Class:

    def __init__(self, N, R, K_total, npop, ngen, U, crosspb, mutpb, rank, size, comm, P, sim_time, tp, algorithm):
        self.N = N
        self.R = R
        self.K_total = K_total
        self.K = [(K_total + i) // R for i in range(R)] #Number of people in network (K = [K1, K2])
        self.mu = np.full((R, N), 1) #Service rate
        self.type_list = np.full(N, 1) #Service type is FCFS (Type1(FCFS),Type2(Processor Sharing: PS),Type3(Infinite Server: IS),Type4(LCFS-PR))
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
        self.scores = [0 for i in range(self.npop)] #各遺伝子のスコア
        self.bestfit_seriese = []#最適遺伝子適合度を入れたリスト
        self.mean_bestfit_seriese = [] #遺伝子全体平均の適合度
        self.algorithm = algorithm #1(MVA), 2(Simulaion)
        self.switch = 2
        #初期遺伝子をブロードキャスト
        prate = 0.2 #人気度の割合
        dim = 2 #拠点間距離の次元数
        if self.rank == 0:
            self.pool = [[self.getRandInt1() for i in range(self.N)] for j in range(self.npop)] #遺伝子を初期化
            if type(tp) is list:
                self.p = tp
            else:
                self.popularity = self.getPopurarity(self.N, self.R, prate) #人気度を設定
                self.distance_matrix = self.getDistance(self.N, dim) #拠点間距離
                self.p = self.getGravity(self.distance_matrix) #推移確率
                np.savetxt('./popularity_std_'+str(self.N)+'_'+str(self.R)+'_'+str(self.K_total)+'_'+str(self.npop)+'_'+str(self.ngen)+'.csv', np.array(self.popularity), delimiter=',', fmt='%d')
                np.savetxt('./distance_std_'+str(self.N)+'_'+str(self.R)+'_'+str(self.K_total)+'_'+str(self.npop)+'_'+str(self.ngen)+'.csv', self.distance_matrix, delimiter=',', fmt='%.5f')
                np.savetxt('./P_std_'+str(self.N)+'_'+str(self.R)+'_'+str(self.K_total)+'_'+str(self.npop)+'_'+str(self.ngen)+'.csv', self.p, delimiter=',')
            if self.algorithm == 2:
                #シミュレーションの初期設定
                self.event = [[] for i in range(self.N)] #各拠点で発生したイベント(arrival, departure)を格納
                self.eventclass = [[] for i in range(self.N)] #各拠点でイベント発生時の客クラス番号
                self.eventqueue = [[] for i in range(self.N)] #各拠点でイベント発生時のqueueの長さ
                self.eventtime = [[] for i in range(self.N)] #各拠点でイベントが発生した時の時刻
                self.queue = np.zeros(self.N) #各拠点のサービス中を含むqueueの長さ(クラスをまとめたもの)
                self.queueclass = np.zeros((self.N, self.R)) #各拠点のサービス中を含むqueueの長さ(クラス別)
                self.classorder = [[] for i in range(self.N)] #拠点に並んでいる順のクラス番号
                self.window = np.full((self.N, self.U), self.R) #サービス中の客クラス(serviceに対応)(self.Rは空状態)
                self.service = np.zeros((self.N, self.U)) #サービス中の客の残りサービス時間
                #開始時の客の分配 (開始時のノードは拠点番号0)
                elapse = 0
                initial_node = 0 #Step1でのみ使用
                for i in range(self.R):
                    for _ in range(self.K[i]):
                        initial_node = random.randrange(self.N)#20220320 最初はランダムにいる拠点を決定
                        self.event[initial_node].append("arrival") #イベントを登録
                        self.eventclass[initial_node].append(i) #到着客のクラス番号
                        self.eventqueue[initial_node].append(self.queue[initial_node])#イベントが発生した時のqueueの長さ(到着客は含まない)
                        self.eventtime[initial_node].append(elapse) #(移動時間0)
                        self.queue[initial_node] += 1 #最初はノード0にn人いるとする
                        self.queueclass[initial_node][i] += 1 #拠点0にクラス別人数を追加
                        self.classorder[initial_node].append(i) #拠点0にクラス番号を追加
                        #空いている窓口に客クラスとサービス時間を登録
                        if self.queue[initial_node] <= self.U:
                            self.window[initial_node][int(self.queue[initial_node] - 1)] = i #クラス番号
                            self.service[initial_node][int(self.queue[initial_node] - 1)] = self.getExponential(self.mu[i][initial_node]) #窓口客のサービス時間設定

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
                    #リストの結合
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
                #世代毎の目的関数値を保存
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
            bcmp_mva = mdl.BCMP_MVA(self.N, self.R, self.K, self.mu, self.type_list, self.p, individual)
            theoretical = bcmp_mva.getMVA()
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
            bcmp_simulation = mdl.BCMP_Simulation(self.N, self.R, self.K, self.U, self.mu, individual, self.type_list, self.p, self.sim_time, self.switch, self.event, self.eventclass, self.eventqueue, self.eventtime, self.queue, self.queueclass, self.classorder, window, service)
            simulation = bcmp_simulation.getSimulation()
            L_class = np.array(simulation) #list to numpy
        np.savetxt('./ga_L_std_'+str(self.N)+'_'+str(self.R)+'_'+str(self.K_total)+'_'+str(self.npop)+'_'+str(self.ngen)+'_'+str(self.U)+'.csv', L_class, delimiter=',')
        np.savetxt('./ga_Node_std_'+str(self.N)+'_'+str(self.R)+'_'+str(self.K_total)+'_'+str(self.npop)+'_'+str(self.ngen)+'_'+str(self.U)+'.csv', individual, delimiter=',') #窓口数
        np.savetxt('./ga_P_std_'+str(self.N)+'_'+str(self.R)+'_'+str(self.K_total)+'_'+str(self.npop)+'_'+str(self.ngen)+'_'+str(self.U)+'.csv', self.p, delimiter=',')
        np.savetxt('./ga_Object_std_'+str(self.N)+'_'+str(self.R)+'_'+str(self.K_total)+'_'+str(self.npop)+'_'+str(self.ngen)+'_'+str(self.U)+'.csv', np.array(self.bestfit_seriese), delimiter=',')
        print('Final Result')
        print('L = {0}'.format(L_class))
        print('sum = {0}'.format(np.sum(L_class)))
        print('Node = {0}'.format(individual))
           
    def getOptimizeBCMP(self, individual):
        if self.algorithm == 1: #MVA
            import BCMP_MVA as mdl
            bcmp_mva = mdl.BCMP_MVA(self.N, self.R, self.K, self.mu, self.type_list, self.p, individual)
            theoretical = bcmp_mva.getMVA()
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
            bcmp_simulation = mdl.BCMP_Simulation(self.N, self.R, self.K, self.U, self.mu, individual, self.type_list, self.p, self.sim_time, self.switch, self.event, self.eventclass, self.eventqueue, self.eventtime, self.queue, self.queueclass, self.classorder, window, service)
            simulation = bcmp_simulation.getSimulation()
            L_class = np.array(simulation) #list to numpy
        L = []
        for i in range(len(L_class)):
            sum = 0
            for j in range(len(L_class[i])):
                sum += L_class[i,j]
            L.append(sum)
        return self.getObjective(L, individual)

   #Create transition probability matrix with gravity model
    def getGravity(self, distance):
        C = 0.1475
        alpha = 1.0
        beta = 1.0
        eta = 0.5
        class_number = len(self.popularity[0]) #Number of classes
        tp = np.zeros((len(distance) * class_number, len(distance) * class_number))
        for r in range(class_number):
            for i in range(len(distance) * r, len(distance) * (r+1)):
                for j in range(len(distance) * r, len(distance) * (r+1)):
                    if distance[i % len(distance)][j % len(distance)] > 0:
                        tp[i][j] = C * (self.popularity[i % len(distance)][r]**alpha) * (self.popularity[j % len(distance)][r]**beta) / (int(distance[i % len(distance)][j % len(distance)])**eta)
        row_sum = np.sum(tp, axis=1)
        for i in range(len(tp)): #Sum of rows to 1
            if row_sum[i] > 0:
                tp[i] /= row_sum[i]
        return tp

    #Objective Function
    def getObjective(self, l, individual):
        l = np.array(l)
        l1 = l.reshape(1,-1)
        val = np.std(l1)
        sum = 0
        for i in range(self.N):
            sum += self.P * (individual[i] - 1)
        return val + sum

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
                    if (bitstring[i]-1) / (self.U-1) > rand():
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
        fig.savefig('./ga_transition_std_'+str(self.N)+'_'+str(self.R)+'_'+str(self.K_total)+'_'+str(self.npop)+'_'+str(self.ngen)+'_'+str(self.U)+'.png')
        
    def getRandInt1(self): #1を返すときに最低利用Number of nodesでの1の返しやすさを反映
        return randint(1, self.U+1)
        
    # 重複なしランダム生成 #https://magazine.techacademy.jp/magazine/21160
    def rand_ints_nodup(self, a, b, k):
        ns = []
        while len(ns) < k:
            n = random.randint(a, b)
            if not n in ns:
                ns.append(n)
        return ns
        
    #人気度ランダム作成関数
    def getPopurarity(self, N, R, prate):
        #Popularity Array
        ranking = [[0 for i in range(R)] for j in range(N)]
        
        #histgram
        histdata = [[],[]] ##
        
        #standard deviation
        scale = 2
        
        #Popular nodes
        for r in range(R):
            pnindex = self.rand_ints_nodup(0, N-1, int(N*prate))
            for n in range(N):
                if n in pnindex:
                    rnd_val = np.random.normal(15, scale) #Normal distribution with mean 15 and standard deviation 1
                    histdata[1].append(rnd_val)
                    ranking[n][r] = round(rnd_val)
                else:
                    rnd_val = np.random.normal(5, scale) #Normal distribution with mean 5 and standard deviation 1
                    histdata[0].append(rnd_val)
                    ranking[n][r] = round(rnd_val)
                
                if ranking[n][r] < 1:
                    ranking[n][r] = 1
        
        return ranking
        
    def getDistance(self, N, dim):
        #Generate location information
        position = np.random.randint(0, 500, (N, dim)) #0~500
        np.savetxt('./position_std_'+str(self.N)+'_'+str(self.R)+'_'+str(self.K_total)+'_'+str(self.npop)+'_'+str(self.ngen)+'_'+str(self.U)+'.csv', position, delimiter=',', fmt='%d')
        
        #Generate distance
        distance = [[-1 for i in range(N)] for j in range(N)] 
        for i in range(N):
            for j in range(N):
                distance[i][j] = np.linalg.norm(position[j]-position[i])
        
        return distance

    def getExponential(self, param): #Set service time
        return - math.log(1 - random.random()) / param


        
if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    algorithm = 2 #1(MVA), 2(Simulaion)
    sim_time = 1000 #Simulation time

    N = int(sys.argv[1]) #Number of nodes
    R = int(sys.argv[2]) #Number of classes
    K_total = int(sys.argv[3]) #Total number of people
    npop = int(sys.argv[4]) #Population size
    ngen = int(sys.argv[5]) #Generation size
    U = int(sys.argv[6]) #Maximum number of windows
    #Set transition probability
    tp_file = 'transition_probability_N33_R2_K500_U2_Core8.csv'
    tp_bool =  os.path.isfile(tp_file)
    if tp_bool == True:
        tp = pd.read_csv(tp_file, index_col=0, header=0).values.tolist()
    else:
        tp = tp_bool
    crosspb = 0.5 #Crossover rate
    mutpb = 0.2 #Mutation rate
    P = 1 #Cost weight

    start = time.time()
    bcmp = BCMP_GA_Class(N, R, K_total, npop, ngen, U, crosspb, mutpb, rank, size, comm, P, sim_time, tp, algorithm)
    best, score = bcmp.genetic_algorithm()
    if rank == 0:
        print('Done!')
        print('f(%s) = %f' % (best, score))
        sum = 0
        for i in range(N):
            sum += P * (best[i] - 1)
        print('std = {0}   cost = {1}'.format(score - sum, sum))
        elapsed_time = time.time() - start
        print ("calclation_time:{0}".format(elapsed_time) + "[sec]")
    

    #mpiexec -n 8 python BCMP_Optimization.py 33 2 500 8 20 2
