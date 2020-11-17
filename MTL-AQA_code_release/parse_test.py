import numpy as np
import matplotlib.pyplot as plt

file_name = "c3d_attn_test_logging_file_1.txt"
file_name2 = "c3davg_test_logging_file_1.txt"
file_name3 = "test_logging_file_1.txt"
file_name4 = "s3d_attn_test_logging_file_1.txt"

experiment_name = "Testing Correlations"

mode = "test" #or test

f = open(file_name, 'r')
lines = f.readlines()


f2 = open(file_name2, 'r')
lines2 = f2.readlines()


f3 = open(file_name3, 'r')
lines3 = f3.readlines()

f4 = open(file_name4, 'r')
lines4 = f4.readlines()

mode_label = None
if mode == 'train':
    mode_label = "Training"
else:
    mode_label = "Testing"

corr1 = []
for i in range(len(lines)/3):
    epoch = lines[i*3:i*3+3]
    corr = float(epoch[1:2][0].split(':')[1])
    corr1.append(corr)

corr2 = []
for i in range(len(lines2)/3):
    epoch = lines2[i*3:i*3+3]
    corr = float(epoch[1:2][0].split(':')[1])
    corr2.append(corr)

corr3 = []
for i in range(len(lines3)/3):
    epoch = lines3[i*3:i*3+3]
    corr = float(epoch[1:2][0].split(':')[1])
    corr3.append(corr)

corr4 = []
for i in range(len(lines4)/3):
    epoch = lines4[i*3:i*3+3]
    corr = float(epoch[1:2][0].split(':')[1])
    corr4.append(corr)

plt.plot(range(len(corr1)), corr1, '-r', label = 'c3d attn')
plt.plot(range(len(corr2)), corr2, '-b', label = 'c3d')
plt.plot(range(len(corr3)), corr3, '-g', label = 's3d')
plt.plot(range(len(corr4)), corr4, '-k', label = 's3d attn')
plt.legend()


plt.title(experiment_name, fontsize=20)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Correlation'.format(mode_label), fontsize=16)
plt.show()