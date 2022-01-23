import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




df = pd.read_csv('results.csv')
df2 = pd.read_csv('results2.csv')
df3 = pd.read_csv('over_flow.csv')
#dimension effect vs sequential
"""
seq_time = np.mean(np.array(df.loc[df['mode'] == 0, 'total_time']).reshape(-1,10), axis=1)
seq_n = np.array(df.loc[df['mode'] == 0, 'n'].unique())
v1_time = np.mean(np.array(df.loc[df['mode'] == 1, 'total_time']).reshape(-1,10), axis=1)
v1_n = np.array(df.loc[df['mode'] == 1, 'n'].unique())
v2_time = np.mean(np.array(df2.loc[df2['mode'] == 2, 'total_time']).reshape(-1,10), axis=1)
v2_n = np.array(df2.loc[df2['mode'] == 2, 'n'].unique())
v3_time = np.mean(np.array(df3.loc[df3['mode'] == 3, 'total_time']).reshape(-1,10), axis=1)
v3_n = np.array(df3.loc[df3['mode'] == 3, 'n'].unique())
plt.plot(seq_n, seq_time, label="Sequential")
plt.plot(v1_n, v1_time, label="Cuda v1")
plt.plot(v2_n, v2_time, label="Cuda v2")
plt.plot(v3_n, v3_time, label="Cuda v3")
plt.xlabel("Matrix dimension", fontsize=16)
plt.ylabel("Execution time(ms)", fontsize=16)
plt.legend(fontsize=16)
plt.title('Sequential vs cuda implemenations in different dimensions', fontsize=20)
plt.show()

#all comparison bar

plt.title('Comparison of all implemenations (Matrix dimensions 5000x5000)', fontsize=20)
plt.bar(range(0, 4), [v1_n[5], v1_time[5], v2_time[5], v3_time[5]], width=0.2)
plt.xticks(range(0, 4), ['Sequential', 'Cuda v1', 'Cuda v2', 'Cuda v3'], fontsize=18)
plt.show()


#speedup bar

plt.title('Speedup (Matrix dimensions 5000x5000)', fontsize=20)
plt.bar(range(0, 3), [v1_n[5]/v1_time[5], v1_n[5]/v2_time[5], v1_n[5]/v3_time[5]], width=0.2)
plt.xticks(range(0, 3), ['Cuda v1', 'Cuda v2', 'Cuda v3'], fontsize=18)
plt.show()
"""

#dimension effect cuda ver

v1_time = np.mean(np.array(df.loc[df['mode'] == 1, 'total_time']).reshape(-1,10), axis=1)
v1_n = np.array(df.loc[df['mode'] == 1, 'n'].unique())
v2_time = np.mean(np.array(df2.loc[df2['mode'] == 2, 'total_time']).reshape(-1,10), axis=1)
v2_n = np.array(df2.loc[df2['mode'] == 2, 'n'].unique())
v3_time = np.mean(np.array(df3.loc[df3['mode'] == 3, 'total_time']).reshape(-1,10), axis=1)
v3_n = np.array(df3.loc[df3['mode'] == 3, 'n'].unique())
plt.plot(v1_n, v1_time, label="Cuda v1")
plt.plot(v2_n, v2_time, label="Cuda v2")
plt.plot(v3_n, v3_time, label="Cuda v3")
plt.xlabel("Dimension", fontsize=16)
plt.ylabel("Execution time(ms)", fontsize=16)
plt.legend(fontsize=18)
plt.title('Cuda implementations comparison for different dimensions', fontsize=20)
plt.show()


"""
#block search plots
df2 = pd.read_csv('b_best_v3.txt')
v2_time = np.mean(np.array(df.loc[df['mode'] == 2, 'process_time']).reshape(-1,10), axis=1)
v2_b = np.array(df.loc[df['mode'] == 2, 'b'].unique())
plt.plot(v2_b, v2_time, label='Cuda v2')
v3_time = np.mean(np.array(df2.loc[df2['mode'] == 3, 'process_time']).reshape(-1,10), axis=1)
v3_b = np.array(df2.loc[df2['mode'] == 3, 'b'].unique())
plt.plot(v3_b, v3_time, label='Cuda v3')
plt.xlabel('block size', fontsize=16)
plt.ylabel('Execution time(ms)', fontsize=16)
plt.title(f"Cuda implementation: block size effect(matrix dimensions = {df.loc[df['mode'] == 3, 'n'].iloc[0]}x{df.loc[df['mode'] == 3, 'n'].iloc[0]})", fontsize=20)
plt.legend(fontsize=16)
plt.show()
print(f'Cuda v2: Min execution time for b={v2_b[np.argmin(v2_time)]}\n')
print(f'Cuda v3: Min execution time for b={v3_b[np.argmin(v3_time)]}\n')
"""

#column major plots
"""
df2 = pd.read_csv('ji.csv')
df = pd.read_csv('results.csv')
df3 = pd.read_csv('results2.csv')
v1_time = np.mean(np.array(df.loc[df['mode'] == 1, 'process_time']).reshape(-1,10), axis=1)
v1_n = np.array(df.loc[df['mode'] == 1, 'n'].unique())
v1_time2 = np.mean(np.array(df2.loc[df2['mode'] == 1, 'process_time']).reshape(-1,10), axis=1)
v1_n2 = np.array(df2.loc[df2['mode'] == 1, 'n'].unique())
v2_time = np.mean(np.array(df3.loc[df3['mode'] == 2, 'process_time']).reshape(-1,10), axis=1)
v2_n = np.array(df3.loc[df3['mode'] == 2, 'n'].unique())
v2_time2 = np.mean(np.array(df2.loc[df2['mode'] == 2, 'process_time']).reshape(-1,10), axis=1)
v2_n2 = np.array(df2.loc[df2['mode'] == 2, 'n'].unique())
plt.title("Cuda v1: Column major vs row major access", fontsize=20)
plt.xlabel("Number of dimension", fontsize=18)
plt.ylabel("Execution time(ms)", fontsize=18)
plt.plot(v1_n, v1_time, label="Cuda v1 (major column)")
plt.plot(v1_n2, v1_time2, label="Cuda v1 (major row)")
plt.legend(fontsize=18)
plt.show()
plt.title("Cuda v2: Column major vs row major access", fontsize=20)
plt.xlabel("Number of dimension", fontsize=18)
plt.ylabel("Execution time(ms)", fontsize=18)
plt.plot(v2_n, v2_time, label="major column access")
plt.plot(v2_n2, v2_time2, label="major row access")
plt.legend(fontsize=18)
plt.show()
"""


#v3 method comparison
"""
df = pd.read_csv('checks.txt')
df2 = pd.read_csv('over_flow.csv')


v3_time_2 = np.mean(np.array(df.loc[df['mode'] == 3, 'total_time']).reshape(-1,10), axis=1)
v3_n_2 = np.array(df.loc[df['mode'] == 3, 'n'].unique())
v3_time = np.mean(np.array(df2.loc[df2['mode'] == 3, 'total_time']).reshape(-1,10), axis=1)
v3_n = np.array(df2.loc[df2['mode'] == 3, 'n'].unique())
print(v3_n, v3_time)
plt.plot(v3_n_2, v3_time_2, label="Cuda v3 (checks)")
plt.plot(v3_n, v3_time, label="Cuda v3 (block with borders)")
plt.xlabel("Matrix dimension", fontsize=16)
plt.ylabel("Execution time(ms)", fontsize=16)
plt.legend(fontsize=16)
plt.title('Comparison of two Cuda v3 impelmentations', fontsize=20)
plt.show()
"""

#memory plots
v2_time1 = np.mean(np.array(df2.loc[df2['mode'] == 2, 'process_time']).reshape(-1,10), axis=1)
v2_time2 = np.mean(np.array(df2.loc[df2['mode'] == 2, 'total_time']).reshape(-1,10), axis=1)
v2_n = np.array(df2.loc[df2['mode'] == 2, 'n'].unique())
plt.plot(v2_n, (v2_time2 - v2_time1), label='Memory Transfer')
plt.plot(v2_n, v2_time1, label='Actual Kernel Execution')
plt.xlabel("Matrix dimension", fontsize=16)
plt.ylabel("Time(ms)", fontsize=16)
plt.title('Memory Transfers vs actual execution', fontsize=20)
plt.legend(fontsize=16)
plt.show()