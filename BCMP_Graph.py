import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
import re

class BCMP_Graph:
    def __init__(self, p, m, L, node_info, dirname, target):
        self.p = p
        self.m = m
        self.L = np.array(L)
        self.node_info = node_info
        self.dirname = dirname

        self.position = self.node_info[['position_x', 'position_y']].values.tolist()
        self.N = len(self.position)             # ノード数
        self.R = self.L.shape[1]                # クラス数（列数）
        self.K_total = sum(m)                   # 合計クラス数
        self.U = max(m) if len(m) > 0 else 0    # 特定ユーザークラス番号
        popularity_cols = [col for col in self.node_info.columns if col.startswith('popularity_')]
        self.sum_popularity = self.node_info[popularity_cols].sum(axis=1).tolist()  # 人気度
        self.size = 1
        
        match = re.search(r'X(\d+)_Y(\d+)', dirname)
        if match:
            self.max_distance_x = int(match.group(1))
            self.max_distance_y = int(match.group(2))
        else:
            raise ValueError("X###_Y### pattern not found in dirname.")
        
        # 1. Heatmap 作成
        heat_scores = [[0 for _ in range(self.max_distance_x)] for _ in range(self.max_distance_y)]
        for r in range(self.R):
            for n in range(self.N):
                for x in range(self.max_distance_x):
                    for y in range(self.max_distance_y):
                        dist = np.linalg.norm(np.array(self.position[n]) - np.array([x, y]))
                        heat_scores[y][x] += self.L[n][r] * (0.99 ** dist)

        maxval = max(map(max, heat_scores))
        fig, ax = plt.subplots()
        ax.contour(range(self.max_distance_x), range(self.max_distance_y), heat_scores,
                   np.linspace(0, maxval, 30), cmap='Blues', vmin=0, vmax=self.K_total)
        contf = ax.contourf(range(self.max_distance_x), range(self.max_distance_y), heat_scores,
                            np.linspace(0, maxval), cmap='Blues', alpha=0.4, vmin=0, vmax=self.K_total)
        ax.set_aspect('equal', 'box')
        plt.colorbar(contf)

        # ノードのプロット
        color = ["c", "g", "m", "b", "r", "w"]

        for ind, p in enumerate(self.position):
            if np.sum(self.L[ind]) != 0:
                #ax.scatter(p[0], p[1], s=10, color='black')
                label_number = ind + 1
                label_text = f"{label_number}"
                if self.sum_popularity[ind] >= 25:
                    ax.scatter(p[0], p[1], s=30, color='red', marker='*')
                else:
                    ax.scatter(p[0], p[1], s=5, color='black')

                ax.text(p[0], p[1], label_text,
                        fontsize=8, ha='center', va='center',
                        fontweight='demibold' if self.m[ind] == self.U else 'normal',
                        color=color[int(self.m[ind]) % len(color)])
                #bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='circle')

        # Heatmap 保存
        plt.savefig(f'{self.dirname}/heatmap_{target}.png',
                    bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

        # 2. Bar chart
        plt.figure(figsize=(12, 5))
        x = np.arange(self.N)
        total_width = 0.8
        for r in range(self.R):
            pos = x - total_width * (1 - (2 * r + 1) / self.R) / 2
            plt.bar(pos, self.L[:, r], label=f'class{r}', width=total_width / self.R)
        plt.legend()
        plt.savefig(f'{self.dirname}/L_{target}.png', dpi=300)
        plt.close()



if __name__ == '__main__':
    dirname = sys.argv[1]
    p = np.loadtxt(f'{dirname}/transition_probability.csv', delimiter=',').tolist()
    m = np.loadtxt(f'{dirname}/m_values.csv', dtype=int)    
    node_info = pd.read_csv(f'{dirname}/node_info.csv')

    targets = ['MVA_L', 'ga_L', 'avg_Lc']
    for target in targets:
        mva_files = glob.glob(os.path.join(dirname, f"{target}*.csv"))
        if not mva_files:
            print(f"No file starting with '{target}' found in the directory.")
            continue
        if target == 'avg_Lc':
            L = pd.read_csv(mva_files[0], index_col=0).values
        else:  
            L = np.loadtxt(mva_files[0], delimiter=',')        
        bcmp = BCMP_Graph(p, m, L, node_info, dirname, target)


#python BCMP_Graph.py N10_R2_K50_U3_X500_Y500 
#python BCMP_Graph.py Optimization_N10_R2_K50_U3_X500_Y500_algorithmSimulation_Size01_20250627194740
