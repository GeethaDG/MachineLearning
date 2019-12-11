import pandas as pd
import numpy as np
import sys
import csv

'''
Reading inputs from Command line
Parameters in command line
1 - File Name - Train_Data_file
2 - Learning Rate n
3 - Threshold  th
'''
cmd = sys.argv[1:]
for each in range(0,len(cmd)):
    if cmd[each] == '--data':
        Train_Data_file = cmd[each+1]
    elif cmd[each] == '--threshold':
        th = float(cmd[each+1])
    elif cmd[each] == '--learningRate':
        n = float(cmd[each+1])
X = pd.read_csv(Train_Data_file, header=None)
Yt = X.loc[:, len(X.columns) - 1]
X.drop(X.columns[len(X.columns) - 1], axis=1, inplace=True)
X.columns = X.columns + 1  # Incrementing the Index
X.insert(0, 0, 1)  # Inserting X0 at Index 0
W = [0] * len(X.columns)  # Initiating W matrix
pre_sse = 0.0
diff_sse = th + 0.1
iteration = 0
out_file=open("Solution.csv", "w")
while diff_sse > th:
    Yp = np.sum(X * W, axis=1)
    error = Yt - Yp
    sse = np.sum(error * error)
    diff_sse = abs(sse - pre_sse)
    pre_sse = sse
    out = str(iteration)
    for i in range(len(W)):
        out = out + ',' + str(round(W[i], 4))
        Wi = W[i]
        Xi = X[i]
        gi = np.sum(Xi * error)
        Wi = Wi + n * gi
        W[i] = Wi
    out = out + ',' + str(round(sse, 4)) + '\n'
    iteration += 1
    out_file.writelines(out)

out_file.close()