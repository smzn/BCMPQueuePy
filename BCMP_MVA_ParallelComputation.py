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
    
    def __init__(self, N, R, K, U, mu, type_list, m, K_total, tp, pid, rank, size, comm):
        self.N = N
        self.R = R
        self.K = K
        self.U = U
        self.K_total = K_total
        self.rank = rank
        self.size = size
        self.comm = comm
        self.pid = pid #Process id
        self.mu = mu #(R×N)
        self.type_list = type_list #Type1(FCFS),Type2(Processor Sharing: PS),Type3(Infinite Server: IS),Type4(LCFS-PR)
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
            if type(tp) is list:
                self.p = tp
            else:
                self.p = self.getTransitionProbability()  
            lmd_list = np.zeros((self.R, (np.max(self.K)+1)**self.R), dtype= float)
            self.Pi_list = np.zeros((self.N, np.max(self.m), (np.max(self.K)+1)**self.R), dtype= float)
            alpha = self.getArrival(self.p)
            print('rank = {0}'.format(self.rank))
        self.lmd_list = comm.bcast(lmd_list, root=0)
        self.alpha = comm.bcast(alpha, root=0) #到着率を共有
        self.km = (np.max(self.K)+1)**self.R
        self.process_text = './process/process_N'+str(self.N)+'_R'+str(self.R)+'_K'+str(self.K_total)+'_U'+str(self.U)+'_Core'+str(self.size)+'.txt'
        self.cpu_list = []
        self.mem_list = []
        self.roop_time = []
        self.combi_len = []
        self.start = time.time()
        

    def getMVA(self):
        state_list = [] #ひとつ前の階層の状態番号
        l_value_list =[] #ひとつ前の状態に対するLの値
        state_dict = {} #ひとつ前の状態に対する{l_value_list, state_list}の辞書
        last_L = []
        #remainder_list = []
        for k_index in range(1, self.K_total+1):
            if self.rank == 0:
                with open(self.process_text, 'a') as f:
                    process = psutil.Process(pid=self.pid)
                    cpu = psutil.cpu_percent(interval=1)
                    print('k = {0}, Memory = {1}MB, CPU = {2}, elapse = {3}'.format(k_index, process.memory_info().rss/1024**2, cpu, time.time() - self.start), file=f) #mem.used/1024**3
                    self.cpu_list.append(cpu)
                    self.mem_list.append(process.memory_info().rss/1024**2) #MB
                    self.roop_time.append(time.time() - self.start)
            else:
                process = psutil.Process(pid=self.pid)
                self.mem_list.append(process.memory_info().rss/1024**2) #MB

            k_combi_list_div_all = [[]for i in range(self.size)]
            if self.rank == 0: #rank0だけが組み合わせを作成
                k_combi_list = self.getCombiList4(self.K, k_index)
                self.combi_len.append(len(k_combi_list))
                k_combi_list_div = k_combi_list_div_all[0]
                quotient, remainder = divmod(len(k_combi_list), self.size) #商とあまり
                with open(self.process_text, 'a') as f:
                    print('Combination = {0}'.format(len(k_combi_list)), file=f)
                    print(k_combi_list, file=f)
                #k_combi_listをsize分だけ分割
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
            
            
            #並列ループ内での受け渡しに利用
            L = np.zeros((self.N, self.R, len(k_combi_list_div)), dtype= float) #平均系内人数
            T = np.zeros((self.N, self.R, len(k_combi_list_div)), dtype= float) #平均系内時間
            lmd = np.zeros((self.R, len(k_combi_list_div)), dtype= float) #各クラスのスループット
            Pi = np.zeros((self.N, np.max(self.m), len(k_combi_list_div)*self.R), dtype= float)

            for idx, val in enumerate(k_combi_list_div):#自分の担当だけ実施
                #Tの更新
                k_state_number = int(self.getState(val)) #kの状態を10進数に変換
                for n in range(self.N): #P336 (8.43)
                    for r in range(self.R):
                        if self.type_list[n] == 3:
                            T[n,r, idx] = 1 / self.mu[r,n]
                        else:
                            r1 = np.zeros(self.R) #K-1rを計算
                            r1[r] = 1 #対象クラスのみ1
                            k1v = val - r1 #ベクトルの引き算

                            #マスタープロセス(rank0)から再帰計算に使う値を取得
                            if self.m[n] > 1:
                                pi_list = [[] for i in range(self.size)] #再帰計算で用いる値を格納
                                if self.rank == 0:
                                    rank_list = [] #計算しているrankを取得
                                    end = remainder
                                    if quotient > idx:
                                        end = self.size #全プロセスで計算を行う場合
                                    #計算を行うrankを取得、rank_listに格納
                                    for rank in range(1, end):
                                        rank_number = self.comm.recv(source=rank, tag=15)
                                        rank_list.append(rank_number)
                                    if len(rank_list) > 0: #pi_listに値を登録
                                        for rank in rank_list:
                                            if rank > 0:
                                                rank_idx = self.comm.recv(source=rank, tag=16)
                                                for i in range(len(rank_idx[0])):
                                                    pi_list[rank].append(self.Pi_list[n][int(rank_idx[0][i])][int(rank_idx[1][i])]) #self.Pi_list[n][j-1][kr_state]
                                        for rank in range(1, end):
                                            self.comm.send(pi_list, dest=rank, tag=19)
                                else:
                                    #対象の値のindexをrank0に送信
                                    if np.min(k1v) >= 0 and np.sum(k1v) > 0:
                                        self.comm.send(self.rank, dest=0, tag=15) #計算を行った場合、rank番号を送信
                                        pi_idx = [[] for i in range(2)] #state_numberとj-1を格納
                                        for j in range(1, self.m[n]):
                                            for _r in range(self.R):
                                                kr = np.zeros(self.R)
                                                kr[_r] = 1
                                                kr_state = int(self.getState_kr(k1v, kr)) #k1vはvalの一つ前の状態
                                                if kr_state < 0:
                                                    continue
                                                pi_idx[0].append(j-1)
                                                pi_idx[1].append(kr_state)
                                        self.comm.send(pi_idx, dest=0, tag=16)
                                    else:
                                        self.comm.send(0, dest=0, tag=15) #計算しない場合、0を送信
                                #rank0から値を受け取る
                                if self.rank > 0:
                                    pi_list =  self.comm.recv(source=0, tag=19)

                            if np.min(k1v) >= 0:
                                kr_state_number = int(self.getState(k1v)) #k-1rの格納位置を取得
                                #1つ前のLとその和を取得
                                sum_l = 0
                                for i in range(self.R): #k-1rを状態に変換
                                    l_value = state_dict.get((kr_state_number,n,i)) #state_listで検索して、l_valueを返す
                                    if l_value is not None:
                                        sum_l += l_value 

                                if self.m[n] == 1:
                                    T[n, r, idx] = 1 / self.mu[r,n] * (1 + sum_l)
                                if self.m[n] > 1:
                                    sum_pi = 0
                                    for _j in range(m[n]-2+1):
                                        #k1vは組み合わせ、kr_state_numberはindex
                                        pi = self.getPi(n, _j, k1v, kr_state_number, Pi, idx, r, pi_list)
                                        Pi[n][_j][idx*self.R + r] = pi
                                        if self.rank == 0: #rank0以外はあとで追加する
                                            self.Pi_list[n][_j][kr_state_number] = pi
                                        sum_pi += (self.m[n] - _j - 1) * pi
                                    T[n, r, idx] = 1 / (self.m[n] * self.mu[r,n]) * (1 + sum_l + sum_pi)
                                
                #λの更新
                for r in range(self.R):
                    sum = 0
                    for n in range(self.N):
                        sum += self.alpha[r][n] * T[n,r,idx]
                    if sum == 0:
                        continue
                    if sum > 0:
                        lmd[r,idx] = val[r] / sum
                        if self.rank == 0: #rank0以外はあとで追加する
                            self.lmd_list[r][k_state_number] = lmd[r][idx]

                #Lの更新
                for n in range(self.N):
                    for r in range(self.R):
                        L[n,r,idx] = lmd[r,idx] * T[n,r,idx] * self.alpha[r][n]

            #全体の処理を集約してからブロードキャスト
            state_list = []
            l_value_list =[]
            state_dict = {}
            n_list = []
            r_list = []
            if self.rank == 0:
                for idx, j in enumerate(k_combi_list_div):
                    k_state = int(self.getState(j))
                    for n in range(self.N):#Lの更新
                        for r in range(self.R):
                            state_list.append(k_state)
                            l_value_list.append(L[n,r,idx])
                            n_list.append(n)
                            r_list.append(r)
                for i in range(1, self.size):
                    lmd_rank = self.comm.recv(source=i, tag=11)
                    l_rank = self.comm.recv(source=i, tag=12)
                    Pi_rank = self.comm.recv(source=i, tag=13)
                    #リストの結合
                    for idx, j in enumerate(k_combi_list_div_all[i]):
                        k_state = int(self.getState(j)) #kの状態を10進数に変換
                        for r in range(self.R): #Lambdaの更新
                            self.lmd_list[r,k_state] = lmd_rank[r,idx]    
                        for n in range(self.N):#Lの更新
                            for r in range(self.R):
                                state_list.append(k_state)
                                l_value_list.append(l_rank[n,r,idx])
                                n_list.append(n)
                                r_list.append(r)
                        #Pi_listの結合
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
            self.comm.barrier() #プロセス同期
            
            if self.rank == 0: #集約完了
                with open(self.process_text, 'a') as f:
                    print('k = {0}, aggregation, elapse = {1}'.format(k_index, time.time() - self.start), file=f)
            
            #ここでブロードキャストする
            self.lmd_list = self.comm.bcast(self.lmd_list, root=0)
            state_list = self.comm.bcast(state_list, root=0)
            l_value_list = self.comm.bcast(l_value_list, root=0)
            if k_index == self.K_total:
                last_L = l_value_list
            n_list = self.comm.bcast(n_list, root=0)
            r_list = self.comm.bcast(r_list, root=0)
            # 辞書に直す
            state_dict = dict(zip(zip(state_list, n_list, r_list),l_value_list))

            if self.rank == 0: #ブロードキャスト完了
                with open(self.process_text, 'a') as f:
                    print('k = {0}, broadcast, elapse = {1}'.format(k_index, time.time() - self.start), file=f)
        
        if self.rank == 0:
            #各プロセスのmem_listの合計
            for i in range(1, self.size):
                mem_rank = self.comm.recv(source=i, tag=14)
                self.mem_list = np.add(self.mem_list, mem_rank)
            df_info = pd.DataFrame({ 'combination': self.combi_len, 'memory' : self.mem_list, 'cpu' : self.cpu_list, 'elapse' : self.roop_time})
            df_info.to_csv('./tp/computation_info_N'+str(N)+'_R'+str(R)+'_K'+str(self.K_total)+'_U'+str(self.U)+'_Core'+str(self.size)+'.csv', index=True)
            np.savetxt('./tp/Node_N'+str(self.N)+'_R'+str(self.R)+'_K'+str(self.K_total)+'_U'+str(self.U)+'_Core'+str(self.size)+'.csv', self.m, delimiter=',')
            model = {'N': self.N, 'R' : self.R, 'K_total' : self.K_total, 'U' : self.U, 'Core' : self.size, 'calclation_time' : time.time() - self.start, 'avg_memory' : np.mean(self.mem_list), 'max_memory' : np.max(self.mem_list)}
            df_info = pd.DataFrame(model, index=['val'])
            df_info.to_csv('./tp/model_info_N'+str(N)+'_R'+str(R)+'_K'+str(self.K_total)+'_U'+str(self.U)+'_Core'+str(self.size)+'.csv', index=True)
        else:
            self.comm.send(self.mem_list, dest=0, tag=14)
        self.comm.barrier() #プロセス同期

        return last_L


    def getPi(self, n, j, k, k_state, Pi, idx, r, pi_list):
        if j == 0 and sum(k) == 0: #Initializationより
            return 1
        if j > 0 and sum(k) == 0: #Initializationより
            return 0
        if j == 0 and sum(k) > 0: #(8.45)
            sum_emlam = 0
            for _r in range(self.R):
                sum_emlam += self.alpha[_r][n] / self.mu[_r][n] * self.lmd_list[_r][k_state]
            sum_pi = 0
            i = 0
            for _j in range(1, self.m[n]):
                pi8_44, i = self.getPi8_44(n, _j, k, k_state, pi_list, i) #(8.44)
                Pi[n][_j][idx*self.R + r] = pi8_44
                if self.rank == 0: #rank0以外はあとで追加する
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
                    sum_val += self.alpha[_r][n] / self.mu[_r][n] * self.lmd_list[_r][k_state] * pi_list[self.rank][i] #pi_list[self.rank][i] #pi_rankの添え字を変える
                    i += 1
        return 1 / j * (sum_val), i #Pi[n][j][idx*self.R + r]

    def getState(self, k):#k=[k1,k2,...]を引数としたときにn進数を返す(R = len(K))
        k_state = 0
        for i in range(self.R): #Lを求めるときの、kの状態を求める(この例では3進数)
            k_state += k[i]*((np.max(self.K)+1)**int(self.R-1-i))
        return k_state

    def getState_kr(self, k, kr):#Piの1つ前の状態
        kr_state = 0
        kkr = k - kr
        if min(kkr) < 0:
            return -1
        else:
            for i in range(self.R):
                kr_state += kkr[i]*((np.max(self.K)+1)**int(self.R-1-i))
            return kr_state

    def getArrival(self, p):#マルチクラスの推移確率からクラス毎の到着率を計算する
        p = np.array(p) #リストからnumpy配列に変換(やりやすいので)
        alpha = np.zeros((self.R, self.N))
        for r in range(self.R): #マルチクラス毎取り出して到着率を求める
            alpha[r] = self.getCloseTraffic(p[r * self.N : (r + 1) * self.N, r * self.N : (r + 1) * self.N])
        return alpha
    
    #閉鎖型ネットワークのノード到着率αを求める
    #https://mizunolab.sist.ac.jp/2019/09/bcmp-network1.html
    def getCloseTraffic(self, p):
        e = np.eye(len(p)-1) #次元を1つ小さくする
        pe = p[1:len(p), 1:len(p)].T - e #行と列を指定して次元を小さくする
        lmd = p[0, 1:len(p)] #0行1列からの値を右辺に用いる
        try:
            slv = solve(pe, lmd * (-1)) #2021/09/28 ここで逆行列がないとエラーが出る
        except np.linalg.LinAlgError as err: #2021/09/29 Singular Matrixが出た時は、対角成分に小さい値を足すことで対応 https://www.kuroshum.com/entry/2018/12/28/python%E3%81%A7singular_matrix%E3%81%8C%E8%B5%B7%E3%81%8D%E3%82%8B%E7%90%86%E7%94%B1
            print('Singular Matrix')
            pe += e * 0.00001 
            slv = solve(pe, lmd * (-1)) 
        alpha = np.insert(slv, 0, 1.0) #α1=1を追加
        return alpha    
    
    def getCombiList4(self, K, Pnum): #並列計算用：Pnumを増やしながら並列計算(2022/1/19)
        #Klist各拠点最大人数 Pnum足し合わせ人数
        Klist = [[j for j in range(K[i]+1)] for i in range(len(K))]
        combKlist = list(itertools.product(*Klist))
        combK = [list(cK) for cK in combKlist if sum(cK) == Pnum ]
        return combK

	#重力モデルで推移確率行列を作成 
    def getGravity(self, distance): #distanceは距離行列、popularityはクラス分の人気度
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
                        tp[i][j] = C * (self.popularity[i % len(distance)][r]**alpha) * (self.popularity[j % len(distance)][r]**beta) / (distance[i % len(distance)][j % len(distance)]**eta)
        row_sum = np.sum(tp, axis=1) #行和を算出
        for i in range(len(tp)): #行和を1にする
            if row_sum[i] > 0:
                tp[i] /= row_sum[i]
        return tp

    def getTransitionProbability(self): #20220313追加
	#(1)拠点の設置と拠点間距離(距離の最大値500)
        node_position_x = [random.randint(0,self.max_distance_x) for i in range(self.N)]
        node_position_y = [random.randint(0,self.max_distance_y) for i in range(self.N)]
        from_id = [] #DF作成用
        to_id = [] #DF作成用
        distance = []
        for i in range(self.N):
            for j in range(i+1,self.N):
                from_id.append(i)
                to_id.append(j)
                distance.append(np.sqrt((node_position_x[i]-node_position_x[j])**2 + (node_position_y[i]-node_position_y[j])**2))
        df_distance = pd.DataFrame({ 'from_id' : from_id, 'to_id' : to_id, 'distance' : distance })#データフレーム化
	    #距離行列の作成
        distance_matrix = np.zeros((self.N, self.N))
        for row in df_distance.itertuples(): #右三角行列で作成される
            distance_matrix[int(row.from_id)][int(row.to_id)] = row.distance
        for i in range(len(distance_matrix)): #下三角に値を入れる(対象)
            for j in range(i+1, len(distance_matrix)):
                distance_matrix[j][i] = distance_matrix[i][j]
        #拠点位置行列の作成
        for i in range(2):
            for j in range(self.N):
                if i == 0:
                    self.position[j][i] = node_position_x[j]
                else:
                    self.position[j][i] = node_position_y[j]
		
	#(2)人気度の設定
        self.popularity = np.abs(np.random.normal(10, 2, (self.N, self.R)))
		
	#(3)推移確率行列の作成
        tp = self.getGravity(distance_matrix)
		
	#(4)拠点情報(拠点番号、位置(x,y)、人気度(Number of classes分))の生成
        df_node = pd.DataFrame({ 'node_number' : range(self.N), 'position_x' : node_position_x, 'position_y' : node_position_y})
        df_node.set_index('node_number', inplace=True)
	    #popularityを追加
        columns = ['popurarity_'+str(i) for i in range(R)]
        for i, val in enumerate(columns):
            df_node[val] = self.popularity[:, i]
		
	#(5)情報の保存
        df_node.to_csv('./tp/node_info_N'+str(N)+'_R'+str(R)+'_K'+str(self.K_total)+'_U'+str(self.U)+'_Core'+str(self.size)+'.csv', index=True)
        df_distance.to_csv('./tp/distance_N'+str(N)+'_R'+str(R)+'_K'+str(self.K_total)+'_U'+str(self.U)+'_Core'+str(self.size)+'.csv', index=True)
        pd.DataFrame(distance_matrix).to_csv('./tp/distance_matrix_N'+str(N)+'_R'+str(R)+'_K'+str(self.K_total)+'_U'+str(self.U)+'_Core'+str(self.size)+'.csv', index=True)
        pd.DataFrame(tp).to_csv('./tp/transition_probability_N'+str(N)+'_R'+str(R)+'_K'+str(self.K_total)+'_U'+str(self.U)+'_Core'+str(self.size)+'.csv', index=True)
        np.savetxt('./tp/popularity_N'+str(self.N)+'_R'+str(self.R)+'_K'+str(self.K_total)+'_U'+str(self.U)+'_Core'+str(self.size)+'.csv', self.popularity, delimiter=',')
        np.savetxt('./tp/position_N'+str(self.N)+'_R'+str(self.R)+'_K'+str(self.K_total)+'_U'+str(self.U)+'_Core'+str(self.size)+'.csv', self.position, delimiter=',')

        return tp

    def plotheatmapL(self, L, position, popularity):
        if type(position) is list: # and tpye(popularity) is list
            self.position = position
            self.popularity = popularity
        #1.contour map
        heat_scores = [[0 for x in range(self.max_distance_x)] for y in range(self.max_distance_y)]
        for r in range(self.R):
            for n in range(self.N):
                for x in range(self.max_distance_x):
                    for y in range(self.max_distance_y):
                        heat_scores[y][x] += (L[n][r])*(0.99**np.linalg.norm(np.array(self.position[n]) - np.array([x,y])))
        maxval = max(list(map(lambda x: max(x), heat_scores)))
        fig, ax = plt.subplots()
        ax.contour(range(self.max_distance_x), range(self.max_distance_y), heat_scores, np.linspace(0,maxval,30), cmap='Blues', vmin=0, vmax=self.K_total)
        contf = ax.contourf(range(self.max_distance_x), range(self.max_distance_y), heat_scores, np.linspace(0,maxval), cmap='Blues', alpha=0.4, vmin=0, vmax=self.K_total)
        ax.set_aspect('equal','box')
        plt.colorbar(contf)
        color = ["c", "g", "m", "b", "r", "w"]
        sum_popularity = np.sum(self.popularity, axis=1)
        for ind, p in enumerate(self.position):
            if (np.sum(L[ind])) != 0:
                ax.scatter(self.position[ind][0], self.position[ind][1], s=1, color='black')
                text = ind
                if sum_popularity[ind] >= 25:
                    text = str(ind) + '*'
                if self.m[ind] == self.U:
                    ax.text(p[0], p[1], "{}".format(text), color=color[int(self.m[ind])], fontweight='demibold')
                else:
                    ax.text(p[0], p[1], "{}".format(text), color=color[int(self.m[ind])])
        plt.savefig('./tp/heatmap_N'+str(self.N)+'_R'+str(self.R)+'_K'+str(self.K_total)+'_U'+str(self.U)+'_Core'+str(self.size)+'.png', bbox_inches = 'tight', pad_inches = 0.1, dpi=300)
        plt.close()
        #2.bar chart
        plt.figure(figsize=(12,5))
        x = np.arange(self.N)
        total_width = 0.8
        for r in range(self.R):
            pos = x - total_width * (1 - (2*r+1)/self.R) / 2
            plt.bar(pos, L[:,r], label='class'+str(r), width = total_width/self.R)
        plt.legend()
        plt.savefig('./tp/L_N'+str(self.N)+'_R'+str(self.R)+'_K'+str(self.K_total)+'_U'+str(self.U)+'_Core'+str(self.size)+'.png', dpi=300)
        plt.close()

        return maxval

    


if __name__ == '__main__':
   
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    pid = os.getpid() #Process id

    N = int(sys.argv[1]) #Number of nodes
    R = int(sys.argv[2]) #Number of classes
    K_total = int(sys.argv[3]) #網内客数
    U = int(sys.argv[4]) #最大窓口数
    K = [(K_total + i) // R for i in range(R)] #クラス人数を自動的に配分する[5, 5]
    mu = np.full((R, N), 1) #サービス率を同じ値で生成(サービス率は調整が必要)
    type_list = np.full(N, 1) #サービスタイプはFCFS
    #窓口数の設定
    m = np.full(N, U) #窓口数(全拠点同一)
    m_file_name = 'Node_N33_R2_K500_U2_Core8.csv'
    tp_file = 'transition_probability_N33_R2_K500_U2_Core8.csv'
    position_file = 'position_N33_R2_K500_U2_Core8.csv'
    popularity_file = 'popularity_N33_R2_K500_U2_Core8.csv'
    m_bool = os.path.isfile(m_file_name) #窓口数のファイルがあればTrue
    if m_bool == True:
        m_file = pd.read_csv(m_file_name, index_col=None, header=None).values.tolist()
        for i in range(N):
            m[i] = int(m_file[i][0])
    #推移確率の設定
    tp_bool =  os.path.isfile(tp_file) #推移確率行列のファイルがあればTrue
    if tp_bool == True:
        tp = pd.read_csv(tp_file, index_col=0, header=0).values.tolist()
        position = pd.read_csv(position_file, index_col=None, header=None).values.tolist()
        popularity = pd.read_csv(popularity_file, index_col=None, header=None).values.tolist()
    else:
        tp = tp_bool
        position = tp_bool
        popularity = tp_bool
    bcmp = BCMP_MVA_Computation(N, R, K, U, mu, type_list, m, K_total, tp, pid, rank, size, comm)
    start = time.time()
    L = bcmp.getMVA()
    elapsed_time = time.time() - start
    print ("calclation_time:{0}".format(elapsed_time) + "[sec]")
    sum = 0
    if rank == 0:
        print('L = \n{0}'.format(L))
        Lr = np.zeros((N, R))
        for n in range(N):
            for r in range(R):
                print('L[{0},{1},{3}] = {2}'.format(n, r, L[(n*R+r)],n*R+r))
                sum += L[(n*R+r)]
                Lr[n, r] = L[(n*R+r)]
        print(sum)
        pd.DataFrame(Lr).to_csv('./tp/L_N'+str(N)+'_R'+str(R)+'_K'+str(K_total)+'_U'+str(U)+'_Core'+str(size)+'.csv', index=True)
        #グラフの作成
        maxval = bcmp.plotheatmapL(Lr, position, popularity)
        

    #mpiexec -n 8 python BCMP_MVA_ParallelComputation.py 33 2 500 2
