import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')
df2 = pd.read_csv('results2.csv')

#dimension effect
seq_time = np.mean(np.array(df.loc[df['mode'] == 0, 'total_time']).reshape(-1,10), axis=1)
seq_n = np.array(df.loc[df['mode'] == 0, 'n'].unique())
v1_time = np.mean(np.array(df.loc[df['mode'] == 1, 'total_time']).reshape(-1,10), axis=1)
v1_n = np.array(df.loc[df['mode'] == 1, 'n'].unique())
v2_time = np.mean(np.array(df2.loc[df2['mode'] == 2, 'total_time']).reshape(-1,10), axis=1)
v2_n = np.array(df2.loc[df2['mode'] == 2, 'n'].unique())
plt.plot(seq_n, seq_time, label="Sequential")
plt.plot(v1_n, v1_time, label="Cuda v1")
plt.plot(v2_n, v2_time, label="Cuda v2")
plt.xlabel("Matrix dimension", fontsize=16)
plt.ylabel("Execution time(ms)", fontsize=16)
plt.legend(fontsize=16)
plt.title('Effect of matrix\'s dimension', fontsize=20)
plt.show()

#block v2, v3
v2_time = np.mean(np.array(df.loc[df['mode'] == 2, 'process_time']).reshape(-1,10), axis=1)
v2_b = np.array(df.loc[df['mode'] == 2, 'b'].unique())
plt.plot(v2_b, v2_time, label='Cuda v2')
v3_time = np.mean(np.array(df.loc[df['mode'] == 3, 'process_time']).reshape(-1,10), axis=1)
v3_b = np.array(df.loc[df['mode'] == 3, 'b'].unique())
plt.plot(v3_b, v3_time, label='Cuda v3')
plt.xlabel('block size', fontsize=16)
plt.ylabel('Execution time(ms)', fontsize=16)
plt.title(f"Cuda implementation: block size effect(matrix dimensions = {df.loc[df['mode'] == 3, 'n'].iloc[0]}x{df.loc[df['mode'] == 3, 'n'].iloc[0]})", fontsize=20)
plt.legend(fontsize=16)
plt.show()
print(f'Cuda v2: Min execution time for b={v2_b[np.argmin(v2_time)]}\n')
print(f'Cuda v3: Min execution time for b={v3_b[np.argmin(v3_time)]}\n')


