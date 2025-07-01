import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import math
import random
import sys
import time
from mpi4py import MPI
import psutil
import os
import copy
import glob
import shutil

class BCMP_Simulation:
    
    def __init__(self, N, R, K, U, mu, m, p, sim_time, switch, output_dir, *args):
        self.output_dir = output_dir
        self.N = N #Number of nodes
        self.R = R #Number of classes
        self.K = K #Number of people (K = [K1, K2])
        self.U = U #Maximum number of windows
        self.mu = mu #service rate
        self.m = m #Number of windows
        self.p = p #transition probability matrix
        self.time = sim_time #simulation time
        self.switch = switch # #Purpose of simulation (1(Verification), 2(Optimization))
        if self.switch == 1:
            self.theoretical = args[0] #exact solution
            self.pid = args[1] #Process id
            self.rank = args[2]
            self.size = args[3]
            self.comm = args[4]
            self.event = [[] for i in range(self.N)] #Stores events (arrival, departure) that occurred at the node
            self.eventclass = [[] for i in range(self.N)] #Class number when event occurs at node
            self.eventqueue = [[] for i in range(N)] #Length of queue when event occurs at node
            self.eventtime = [[] for i in range(N)] #Time when the event occurred at the node
            self.queue = np.zeros(self.N) #Length of queue including in-service of node
            self.queueclass = np.zeros((self.N, self.R)) #Length of queue including in-service of node (by class)
            self.classorder = [[] for i in range(self.N)]  # Class order of customers at each node / 拠点に並んでいる順のクラス番号
            self.window = np.full((self.N, self.U), self.R)  # Classes currently in service at each node (R means empty) / サービス中の客クラス（self.Rは空状態）
            self.service = np.zeros((self.N, self.U))  # Remaining service time of customers in service / サービス中の客の残りサービス時間
            self.rmse = []  # RMSE values recorded every 100 time units / 100単位時間でのrmseの値を格納
            self.rmse_time = []  # Time points when RMSE was recorded / rmseを登録した時間
            self.regist_time = 50  # First time to record RMSE / rmseの登録時刻
            self.regist_span = 50  # Interval between RMSE recordings (every 50 units) / 50単位で登録
            self.length = [[] for i in range(self.N)]  # Change rate of number of customers in the system / 系内人数の変化率
            self.cpu_list = []  # List to store CPU usage / CPU使用率を格納するリスト
            self.mem_list = []  # List to store memory usage / メモリ使用量を格納するリスト
            self.time_list = []  # List to store time stamps / 時刻を格納するリスト
            if rank == 0:
                self.sum_L = np.zeros(self.N)  # Total number in system (for averaging) / 平均系内人数（結果の和）
                self.sum_Lc = np.zeros((self.N, self.R))  # Total number in system by class / 平均系内人数（結果の和）（クラス別）
                self.sum_Q = np.zeros(self.N)  # Total number in queue (for averaging) / 平均待ち人数（結果の和）
                self.sum_Qc = np.zeros((self.N, self.R))  # Total number in queue by class / 平均待ち人数（結果の和）（クラス別）

                
        
        if self.switch == 2:
            self.event = copy.deepcopy(args[0]) #Stores events (arrival, departure) that occurred at the node
            self.eventclass = copy.deepcopy(args[1]) #Class number when event occurs at node
            self.eventqueue = copy.deepcopy(args[2]) #Length of queue when event occurs at node
            self.eventtime = copy.deepcopy(args[3]) #Time when the event occurred at the node
            self.queue = copy.deepcopy(args[4]) #Length of queue including in-service of node
            self.queueclass = copy.deepcopy(args[5]) #Length of queue including in-service of node (by class)
            self.classorder = copy.deepcopy(args[6]) #Class number in the order in which they are lined up in the node
            self.window = copy.deepcopy(args[7]) #Class number in service (self.R is empty)
            self.service = copy.deepcopy(args[8]) #Remaining service time
        

        self.L = np.zeros(self.N) #Average number of people in the node
        self.Lc = np.zeros((self.N, self.R)) #Average number of people in the node (by class)
        self.Q = np.zeros(self.N) #平均待ち人数(結果)
        self.Qc = np.zeros((self.N, self.R)) #平均待ち人数(結果)(クラス別)
        self.timerate = np.zeros((self.N, sum(self.K)+1)) #Distribution of the number of people at a node (distribution of 0~K people)
        self.timerateclass = np.zeros((self.N, self.R, sum(self.K)+1)) #Distribution of the number of people at a node (distribution of 0~K people) (by class)
        self.start = time.time()

        
    def getSimulation(self):
        window_number = np.zeros((self.N, self.U))
        total_length = np.zeros(self.N) #Total number of people in the node
        total_lengthclass = np.zeros((self.N, self.R)) #Total number of people in the node (by class)
        total_waiting = np.zeros(self.N) #Total number of people waiting
        total_waitingclass = np.zeros((self.N, self.R)) #Total number of people waiting (by class)

        elapse = 0

        if self.switch == 1:
            # Step 1: Distribute customers at the start (initial node is node 0) / Step1 開始時の客の分配（開始時のノードは拠点番号0）
            initial_node = 0
            for i in range(self.R):
                for j in range(self.K[i]):
                    initial_node = random.randrange(self.N)  # Randomly assign initial node to each customer / 最初はランダムにいる拠点を決定
                    self.event[initial_node].append("arrival")  # Register arrival event / イベントを登録
                    self.eventclass[initial_node].append(i)  # Record customer class for arrival / 到着客のクラス番号
                    self.eventqueue[initial_node].append(self.queue[initial_node])  # Queue length at event (excluding arriving customer) / イベントが発生した時のqueueの長さ（到着客は含まない）
                    self.eventtime[initial_node].append(elapse)  # Event time (no travel time) / イベント時間（移動時間0）
                    self.queue[initial_node] += 1  # Increase total number of customers at node / 最初はノードに1人追加
                    self.queueclass[initial_node][i] += 1  # Increase number of class-i customers at node / 拠点にクラス別人数を追加
                    self.classorder[initial_node].append(i)  # Record class order at the node / 拠点にクラス番号を追加
                    if self.queue[initial_node] <= self.m[initial_node]:  # If there is an open service window / 窓口が空いている状態なら
                        self.window[initial_node][int(self.queue[initial_node] - 1)] = i  # Assign class to service window / 窓口にクラス番号を登録
                        self.service[initial_node][int(self.queue[initial_node] - 1)] = self.getExponential(self.mu[i][initial_node])  # Set service time / 窓口客のサービス時間設定

        #Set window_number
        for i in range(self.N):
            num = 1
            for j in range(self.m[i]):
                if self.window[i][j] < self.R:
                    window_number[i][j] = num
                    num += 1


        #Step2 Simulation Start
        while elapse < self.time:
            mini_service = 100000 #Minimum service time
            mini_index = -1 #node index
            window_index = -1
            classorder_index = 0


            #Step2.1 Search for the node where the next eviction will occur
            for i in range(self.N):
                if self.queue[i] > 0:
                    for j in range(self.m[i]):
                        if self.window[i][j] < self.R:
                            if mini_service > self.service[i][j]: #Update minimum service
                                mini_service = self.service[i][j]
                                mini_index = i
                                window_index = j
            for i in range(self.m[mini_index]):
                if window_number[mini_index][i] == window_index + 1:
                    classorder_index = i #node index
            departure_class = self.classorder[mini_index].pop(classorder_index)
            #Update window_number
            for i in range(classorder_index, self.m[mini_index]):
                if i+1 == self.U:
                    window_number[mini_index][i] = 0
                else:
                    window_number[mini_index][i] = window_number[mini_index][i+1]
    

            #Step2.2 Update information on all nodes (service time, total number of people)
            for i in range(self.N):
                total_length[i] += self.queue[i] * mini_service #Total number of people in the node
                for r in range(self.R):
                    total_lengthclass[i,r] += self.queueclass[i,r] * mini_service #by class
                if self.queue[i] > 0:
                    for j in range(self.m[i]):
                        if self.service[i][j] > 0:
                            self.service[i][j] -= mini_service #Reduce service time
                    if self.queue[i] > self.m[i]:
                        total_waiting[i] += ( self.queue[i] - self.m[i] ) * mini_service
                    else:
                        total_waiting[i] += 0 * mini_service
                    for r in range(self.R):
                        if self.queueclass[i,r] > 0:
                            c = np.count_nonzero(self.window[i]==r)
                            total_waitingclass[i,r] += ( self.queueclass[i,r] - c ) * mini_service 
                elif self.queue[i] == 0:
                    total_waiting[i] += self.queue[i] * mini_service
                self.timerate[i, int(self.queue[i])] += mini_service
                for r in range(self.R):
                    self.timerateclass[i, r, int(self.queueclass[i,r])] += mini_service
            
        
            #Step2.3 Reflects events
            self.event[mini_index].append("departure")
            self.eventclass[mini_index].append(departure_class)
            self.eventqueue[mini_index].append(self.queue[mini_index])
            self.queue[mini_index] -= 1
            self.queueclass[mini_index, departure_class] -= 1
            elapse += mini_service
            self.eventtime[mini_index].append(elapse)
            self.window[mini_index][window_index] = self.R
            #If there is a waiting list, Service start
            if self.queue[mini_index] > 0:
                for i in range(self.m[mini_index]):
                    if self.service[mini_index][i] == 0 and self.window[mini_index][i] == self.R and self.queue[mini_index] >= self.m[mini_index]:
                        window_number[mini_index][int(self.m[mini_index] - 1)] = i+1
                        self.window[mini_index][i] = self.classorder[mini_index][int(self.m[mini_index] - 1)]
                        self.service[mini_index][i] = self.getExponential(self.mu[departure_class][mini_index]) #Set service time
                        break

            
            #Step2.4 Decide where to go
            rand = random.random()
            sum_rand = 0
            destination_index = -1
            pr = np.zeros((self.N, self.N))
            #Obtain the transition probability matrix of the target
            for i in range(self.N * departure_class, self.N * (departure_class + 1)):
                for j in range(self.N * departure_class, self.N * (departure_class + 1)):
                    pr[i - self.N * departure_class, j - self.N * departure_class] = self.p[i][j]
            for i in range(len(pr)):
                sum_rand += pr[mini_index][i]
                if rand < sum_rand:
                    destination_index = i
                    break
            if destination_index == -1:
                destination_index = len(pr) -1
            
            #Reflects events
            self.event[destination_index].append("arrival")
            self.eventclass[destination_index].append(departure_class)
            self.eventqueue[destination_index].append(self.queue[destination_index])
            self.eventtime[destination_index].append(elapse)
            self.queue[destination_index] += 1
            self.queueclass[destination_index][departure_class] += 1 
            self.classorder[destination_index].append(departure_class)
            #If there is an open window, Service start
            if self.queue[destination_index] <= self.m[destination_index]:
                for i in range(self.m[destination_index]):
                    if self.service[destination_index][i] == 0 and self.window[destination_index][i] == self.R:
                        window_number[destination_index][int(self.queue[destination_index] - 1)] = i+1
                        self.window[destination_index][i] = departure_class
                        self.service[destination_index][i] = self.getExponential(self.mu[departure_class][destination_index]) #Set service time
                        break


            if self.switch == 1:
                # Step 2.5: Calculate RMSE / Step2.5 RMSEの計算
                if elapse > self.regist_time:
                    rmse_sum = 0
                    lc = total_lengthclass / elapse  # Average number in system per class over elapsed time / 今までの時刻での平均系内人数（クラス別）
                    for n in range(self.N):
                        for r in range(self.R):
                            rmse_sum += (self.theoretical[n][r] - lc[n][r])**2
                    rmse_sum /= self.N * self.R
                    rmse_value = math.sqrt(rmse_sum)
                    self.rmse.append(rmse_value)
                    self.rmse_time.append(self.regist_time)
                    self.regist_time += self.regist_span
                    calculation_time = time.time() - self.start
                    if self.rank == 0:
                        process = psutil.Process(pid=self.pid)
                        cpu = psutil.cpu_percent(interval=1)
                        self.cpu_list.append(cpu)
                        self.mem_list.append(process.memory_info().rss/1024**2) #MB
                        self.time_list.append(calculation_time)
                    else:
                        process = psutil.Process(pid=self.pid)
                        self.mem_list.append(process.memory_info().rss/1024**2) #MB
                        self.time_list.append(calculation_time)
                    # Number of customers in the system over time / 時間経過による系内人数
                    for n in range(self.N):
                        self.length[n].append(self.queue[n])

        
        self.L = total_length / self.time #Mean number of people in the node
        self.Lc = total_lengthclass / self.time #Mean number of people in the node (by class)
        self.Q = total_waiting / self.time  # Average number of customers waiting / 平均待ち人数
        self.Qc = total_waitingclass / self.time  # Average number of customers waiting per class / 平均待ち人数（クラス別）


        if self.switch == 1:
            pd.DataFrame(self.mem_list).to_csv(f'./{self.output_dir}/memory_Rank_'+str(self.rank)+'.csv')        
            pd.DataFrame(self.L).to_csv(f'./{self.output_dir}/L_Rank_'+str(self.rank)+'.csv')
            pd.DataFrame(self.Lc).to_csv(f'./{self.output_dir}/Lc_Rank_'+str(self.rank)+'.csv')
            pd.DataFrame(self.Q).to_csv(f'./{self.output_dir}/Q_Rank_'+str(self.rank)+'.csv')
            pd.DataFrame(self.Qc).to_csv(f'./{self.output_dir}/Qc_Rank_'+str(self.rank)+'.csv')
            rmse_index = {'time': self.rmse_time, 'RMSE': self.rmse}
            df_rmse = pd.DataFrame(rmse_index)
            df_rmse.to_csv(f'./{self.output_dir}/RMSE_Rank_'+str(self.rank)+'.csv')
            time_index = {'simulation_time': self.rmse_time, 'calculation_time': self.time_list}
            time_index = {'calculation_time': self.time_list}
            df_time = pd.DataFrame(time_index)
            df_time.to_csv(f'./{self.output_dir}/Time_Rank_'+str(self.rank)+'.csv')
            if self.rank == 0:
                sum_mem = self.mem_list
                sum_rmse = []
                sum_rmse.append(self.rmse[-1])
                for i in range(1, self.size):
                    mem_rank = self.comm.recv(source=i, tag=10)
                    rmse_rank = self.comm.recv(source=i, tag=11)
                    # Sum of mem_list from all processes / 各プロセスのmem_listの合計
                    sum_mem = np.add(sum_mem, mem_rank)
                    sum_rmse.append(rmse_rank)
                if len(sum_rmse) > 2:
                    max_index = np.argmax(sum_rmse)
                    sum_rmse.pop(max_index)
                    # Recalculate min_index after popping the max value / max を pop したので min_index を再計算
                    min_index = np.argmin(sum_rmse)
                    sum_rmse.pop(min_index)
                model = {'N': self.N, 'R' : self.R, 'K_total' : sum(self.K), 'U' : self.U, 'Core' : self.size, 'calclation_time' : time.time() - self.start, 'avg_memory' : np.mean(self.mem_list), 'max_memory' : np.max(self.mem_list), 'avg_cpu' : np.mean(self.cpu_list), 'max_cpu' : np.max(self.cpu_list)}
                df_info = pd.DataFrame(model, index=['val'])
                df_info.to_csv(f'./{self.output_dir}/model_info.csv', index=True)
            else:
                self.comm.send(self.mem_list, dest=0, tag=10)
                self.comm.send(self.rmse[-1], dest=0, tag=11)


        return self.Lc
        

    def getGraph(self):
        plt.figure(figsize=(12,5))
        tmp = list(matplotlib.colors.CSS4_COLORS.values())
        colorlist = tmp[:self.N] # First N colors / 先頭からN個
        # 系内人数の変動 / Change in number of customers in the system
        for n in range(self.N):
            plt.plot(self.rmse_time, self.length[n], '-', lw=0.5, color=colorlist[n], label='node'+str(n))
        plt.legend(fontsize='xx-small', ncol=3, bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.1)
        plt.savefig(f'./{self.output_dir}/length_Rank_'+str(self.rank)+'.png', format='png', dpi=300,  bbox_inches='tight')
        plt.close()

        # 各rankのRMSEの最終値を取得 / Get final RMSE value from each rank
        if self.rank == 0:
            last_rmse = []
            last_rmse.append(self.rmse[-1])

            for i in range(1, self.size):
                rmse_rank = self.comm.recv(source=i, tag=12)
                last_rmse.append(rmse_rank[-1])
        else:
            self.comm.send(self.rmse, dest=0, tag=12)
        self.comm.barrier() # プロセス同期 / Synchronize all processes
        # RMSEの最終値が最小と最大のものを除いて、平均系内人数の平均を算出 / Exclude min/max RMSE ranks and compute average number in system
        # 各シミュレーション時間におけるRMSEの平均の算出 / Compute average RMSE over all simulation time steps
        if self.rank == 0:
            # 平均算出用 / For average calculation
            sum_L = np.zeros(self.N)  # Total average number in system / 平均系内人数（結果の和）
            sum_Lc = np.zeros((self.N, self.R))  # Total average number in system by class / 平均系内人数（結果の和）（クラス別）
            sum_Q = np.zeros(self.N)  # Total average queue length / 平均待ち人数（結果の和）
            sum_Qc = np.zeros((self.N, self.R))  # Total average queue length by class / 平均待ち人数（結果の和）（クラス別）
            avg_rmse = np.zeros(len(self.rmse_time))  # Average RMSE across time / RMSEの平均
            # RMSEの総和が最大と最小のrank取得 / Get ranks with max and min final RMSE
            max_index = last_rmse.index(max(last_rmse))
            min_index = last_rmse.index(min(last_rmse))

            plt.figure(figsize=(12,5))
            if 0 == max_index or 0 == min_index: # rank0が最大最小の場合 / If rank 0 is either max or min
                plt.plot(self.rmse_time, self.rmse, linestyle = 'dotted', color = 'black', alpha = 0.5, label='Excluded Rank 0') #平均に含まれない
            else:
                sum_L += self.L
                sum_Lc += self.Lc
                sum_Q += self.Q
                sum_Qc += self.Qc
                avg_rmse = np.add(avg_rmse, self.rmse)
                plt.plot(self.rmse_time, self.rmse, label='Rank 0')
            
            for i in range(1, self.size):                                   
                L_rank = self.comm.recv(source=i, tag=13)
                Lc_rank = self.comm.recv(source=i, tag=14)
                Q_rank = self.comm.recv(source=i, tag=15)
                Qc_rank = self.comm.recv(source=i, tag=16)
                rmse_rank = self.comm.recv(source=i, tag=17)
                time_rank = self.comm.recv(source=i, tag=18)
                

                if i == max_index or i == min_index:
                    plt.plot(time_rank, rmse_rank, linestyle = 'dotted', color = 'black', alpha = 0.5, label=f'Excluded Rank {i}') #平均に含まれない
                else:
                    sum_L += L_rank
                    sum_Lc += Lc_rank
                    sum_Q += Q_rank
                    sum_Qc += Qc_rank
                    avg_rmse = np.add(avg_rmse, rmse_rank)
                    plt.plot(time_rank, rmse_rank, label=f'Rank {i}')
            
            plt.legend()
            plt.savefig(f'./{self.output_dir}/RMSE.png', format='png', dpi=300)
            plt.clf()
                    
            # Calculate averages / 平均の算出
            if self.size > 2:           
                avg_L = sum_L / (self.size - 2)
                avg_Lc = sum_Lc / (self.size - 2)
                avg_Q = sum_Q / (self.size - 2)
                avg_Qc = sum_Qc / (self.size - 2)
                avg_rmse = [n / (self.size - 2) for n in avg_rmse] #rmseの平均

                
                pd.DataFrame(avg_L).to_csv(f'./{self.output_dir}/avg_L_Size_'+str(self.size)+'.csv')
                pd.DataFrame(avg_Lc).to_csv(f'./{self.output_dir}/avg_Lc_Size_'+str(self.size)+'.csv')
                pd.DataFrame(avg_Q).to_csv(f'./{self.output_dir}/avg_Q_Size_'+str(self.size)+'.csv')
                pd.DataFrame(avg_Qc).to_csv(f'./{self.output_dir}/avg_Qc_Size_'+str(self.size)+'.csv')
                avg_rmse_index = {'time': self.rmse_time, 'RMSE': avg_rmse}
                df_avg_rmse = pd.DataFrame(avg_rmse_index)
                df_avg_rmse.to_csv(f'./{self.output_dir}/avg_RMSE_Size_'+str(self.size)+'.csv')
                plt.grid()
                plt.plot(self.rmse_time, avg_rmse)
                plt.savefig(f'./{self.output_dir}/avg_RMSE_Size_'+str(self.size)+'.png', format='png', dpi=300)
                plt.clf()
                plt.close()
        
        else:
            self.comm.send(self.L, dest=0, tag=13)
            self.comm.send(self.Lc, dest=0, tag=14)
            self.comm.send(self.Q, dest=0, tag=15)
            self.comm.send(self.Qc, dest=0, tag=16)
            self.comm.send(self.rmse, dest=0, tag=17)
            self.comm.send(self.rmse_time, dest=0, tag=18)
        self.comm.barrier() #

        
    def getExponential(self, param):
        return - math.log(1 - random.random()) / param
    
    
    
if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    pid = os.getpid() #Process id
    switch = 1 #Purpose of simulation (1(Verification), 2(Optimization))
    #2 (Optimization) is only used when called from BCMP_Optimization.
    #2(Optimization)はBCMP_Optimizationから呼び出す時のみ利用

    N = int(sys.argv[1]) #Number of nodes in the network
    R = int(sys.argv[2]) #Number of classes
    K_total = int(sys.argv[3]) #Total number of people in the network
    U = int(sys.argv[4]) #Maximum number of windows
    
    max_x = int(sys.argv[5]) if len(sys.argv) > 5 else 500
    max_y = int(sys.argv[6]) if len(sys.argv) > 6 else 500

    sim_time = int(sys.argv[7]) #Simulation time
        
    dirname = f'../results/N{N}_R{R}_K{K_total}_U{U}_X{max_x}_Y{max_y}'
    p = np.loadtxt(f'{dirname}/transition_probability.csv', delimiter=',').tolist()
    m = np.loadtxt(f'{dirname}/m_values.csv', dtype=int)
    mu = np.loadtxt(f'{dirname}/mu_matrix.csv', delimiter=',')
    K = np.loadtxt(f'{dirname}/K_values.csv', delimiter=',').astype(int).tolist()
    
    # Search for files matching a pattern / パターンにマッチするファイルを探す
    mva_files = glob.glob(os.path.join(dirname, "MVA_L_matrix*.csv"))
    if not mva_files:
        raise FileNotFoundError("No file starting with 'MVA_L_matrix' found in the directory.")
    # Load the first file found / 最初に見つかったファイルを読み込む
    theoretical = np.loadtxt(mva_files[0], delimiter=',').tolist()
    
    if rank == 0:
        output_dir = time.strftime(f"../results/Simulation_N{N}_R{R}_K{K_total}_U{U}_X{max_x}_Y{max_y}_Size{size:02}_%Y%m%d%H%M%S")
        os.makedirs(f'{output_dir}', exist_ok=True)
        
        # Create log file / ログファイル作成
        log_path = os.path.join(output_dir, "run_info.txt")
        with open(log_path, 'w') as f:
            f.write(f"Execution Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"N (Number of nodes): {N}\n")
            f.write(f"R (Number of classes): {R}\n")
            f.write(f"K_total (Total people): {K_total}\n")
            f.write(f"U (Max windows): {U}\n")
            f.write(f"Purpose of simulation: {switch} ({'MVA' if switch == 1 else 'Simulation'})\n")
            f.write(f"Simulation time: {sim_time}\n")
            f.write(f"MPI  size: {size}\n")
            f.write(f"Transition Probability file: {dirname}/transition_probability.csv\n")
            f.write(f"m Matrix file: {dirname}/m_matrix.csv\n")
            f.write(f"mu Matrix file: {dirname}/mu_matrix.csv\n")
            f.write(f"K Values file: {dirname}/K_values.csv\n")
            f.write(f"Theoretical Values file: {dirname}/{mva_files[0]}\n")
            
            
        # List of filenames to be copied / コピー対象のファイル名一覧
        files_to_copy = [
            "node_info.csv",
            "distance.csv",
            "distance_matrix.csv",
            "transition_probability.csv",
            "popularity.csv",
            "position.csv",
            "m_values.csv",
            "mu_matrix.csv",
            "K_values.csv"
        ] + [os.path.basename(mva_files[0])]
       # Copy each file from output_dir to dirname / 各ファイルをoutput_dirからdirnameにコピー
        for filename in files_to_copy:
            src_path = os.path.join(dirname, filename)
            dst_path = os.path.join(output_dir, filename)
            shutil.copy(src_path, dst_path)
    else:
        output_dir = None  # rank != 0 は最初は None
    
    # Broadcast output_dir from rank 0 to all ranks
    output_dir = comm.bcast(output_dir, root=0)
    
    bcmp = BCMP_Simulation(N, R, K, U, mu, m, p, sim_time, switch, output_dir, theoretical, pid, rank, size, comm)
    start = time.time()
    L = bcmp.getSimulation()
    elapsed_time = time.time() - start
    print ("rank : {1}, calclation_time:{0}".format(elapsed_time, rank) + "[sec]")
    bcmp.getGraph()


    #mpiexec -n 8 python BCMP_Simulation.py 10 2 50 3 500 500 10000