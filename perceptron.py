import pandas as pd
import numpy as np
import sys
import csv

'''
Reading inputs from Command line
Parameters in command line
1 - Input File Name - Train_Data_file
2 - Output File Name - Output_file
example : python3 perceptron.py --data Example.tsv --output Example_Errors.tsv
'''
cmd = sys.argv[1:]
for each in range(0,len(cmd)):
    if cmd[each] == '--data':
        Train_Data_file = cmd[each+1]
    elif cmd[each] == '--output':
        Output_file = cmd[each+1]

#Reading input tsv file and preprocessing
X = pd.read_csv(Train_Data_file,delimiter='\t',header=None)
rows, cols = X.shape
for i in range(cols):
    if X[i].isna().sum()!=0 and X[i].isna().sum()==rows:
            X.drop(X.columns[i], axis=1, inplace=True)
i=0
Yt = pd.DataFrame(X.loc[:, 0])
#Mapping class variables with binary values
Yt=Yt.replace(to_replace = 'A', value = int(1))
Yt=Yt.replace(to_replace = 'B', value = int(0))
X.drop(X.columns[0], axis=1, inplace=True)
X.insert(0, 0, 1)
W = [0] * len(X.columns) #weights for constant learing rate
Wt = [0] * len(X.columns) #weights for annealing learning rate
out_error = pd.DataFrame([0] * 100)
n=1 #learning rate
iteration = 0
while iteration<=100:
    iteration += 1
    Yp = pd.DataFrame(np.sum(X * W, axis=1))  #constant learning rate output
    Yp[1]=np.where(Yp[0] > 0, int(1) , int(0))  #constant learning rate output normalised
    Yp[2]=pd.DataFrame(np.sum(X * Wt, axis=1)) #annealing learning rate output
    Yp[3]=np.where(Yp[2] > 0, int(1), int(0)) #annealing learning rate output normalised
    error=pd.DataFrame(Yt.loc[:, 0]-Yp.loc[:, 1])  #constant learning rate error
    error[2]=pd.DataFrame(Yt.loc[:, 0]-Yp.loc[:, 3]) #annealing learning rate error
    for i in range(len(W)):
        Wi = W[i]
        Wti = Wt[i]
        Xi = X[i]
        gi = np.sum(Xi * error.loc[:,0])
        git = np.sum(Xi * error.loc[:,2])
        Wi = Wi + n * gi
        Wti = Wti + (n/iteration) * git
        W[i] = Wi
        Wt[i] = Wti
    error[1] = np.where(error[0]!=0,int(1),int(0)) #constant learning rate error normalised
    error[3] = np.where(error[2] != 0, int(1), int(0)) #annealing learning rate error normalised
    out_error.loc[iteration-1,0] = str(int(np.sum(error.loc[:, 1]**2))) #constant learning rate misclassifications
    out_error.loc[iteration-1,1] = str(int(np.sum(error.loc[:, 3]**2))) #annealing learning rate misclassifications


#Printing error into a .tsv file
with open(Output_file, "w", newline="") as out_file:
    writer = csv.writer(out_file, delimiter='\t')
    writer.writerow(out_error.loc[:,0])
    writer.writerow(out_error.loc[:,1])

out_file.close()










