import numpy as np
import matplotlib.pyplot as plt

file_name = "c3d_attn_train_logging_file_1.txt"
file_name2 = "c3davg_train_logging_file_1.txt"
file_name3 = "train_logging_file_1.txt"
file_name4 = "s3d_attn_train_logging_file_1.txt"
file_name5 = "c3davg_8_gru_attn_train_logging_file_1.txt"

experiment_name = "Training Losses"

mode = "train" #or test

f = open(file_name, 'r')
lines = f.readlines()

f2 = open(file_name2, 'r')
lines2 = f2.readlines()
print(len(lines2))

f3 = open(file_name3, 'r')
lines3 = f3.readlines()

f4 = open(file_name4, 'r')
lines4 = f4.readlines()

f5 = open(file_name5, 'r')
lines5 = f5.readlines()

mode_label = None
if mode == 'train':
    mode_label = "Training"
else:
    mode_label = "Testing"

num_epochs = len(lines)/12
print(num_epochs)

losses = []
fs_losses = []
cls_losses = []
cap_losses = []

for i in range(len(lines)/12):

    epoch = lines[i*12:i*12+12]
    first_iter = epoch[0:4]
    second_iter = epoch[4:8]
    last_iter = epoch[8:]

    first_loss = float(first_iter[0].split(',')[2].split(':')[1])
    middle_loss = float(second_iter[0].split(',')[2].split(':')[1])


    first_line = last_iter[0]
    first_line = first_line.split(',')
    first_line = [i.split(':') for i in first_line[2:]]
    loss = float(first_line[0][1])
    fs_loss = float(first_line[1][1])

    cls_loss = float(last_iter[-3].split(':')[1])
    cap_loss = float(last_iter[-2].split(':')[1])

    # losses.append(first_loss)
    # losses.append(middle_loss)
    losses.append(loss)
    fs_losses.append(fs_loss)
    cls_losses.append(cls_loss)
    cap_losses.append(cap_loss)

losses_2 = []


for i in range(len(lines2)/12):

    epoch = lines2[i*12:i*12+12]
    first_iter = epoch[0:4]
    second_iter = epoch[4:8]
    last_iter = epoch[8:]

    first_loss = float(first_iter[0].split(',')[2].split(':')[1])
    middle_loss = float(second_iter[0].split(',')[2].split(':')[1])


    first_line = last_iter[0]
    first_line = first_line.split(',')
    first_line = [i.split(':') for i in first_line[2:]]
    loss = float(first_line[0][1])



    # losses.append(first_loss)
    # losses.append(middle_loss)
    losses_2.append(loss)


losses_3 = []


for i in range(len(lines3)/12):

    epoch = lines3[i*12:i*12+12]
    first_iter = epoch[0:4]
    second_iter = epoch[4:8]
    last_iter = epoch[8:]

    first_loss = float(first_iter[0].split(',')[2].split(':')[1])
    middle_loss = float(second_iter[0].split(',')[2].split(':')[1])


    first_line = last_iter[0]
    first_line = first_line.split(',')
    first_line = [i.split(':') for i in first_line[2:]]
    loss = float(first_line[0][1])



    # losses.append(first_loss)
    # losses.append(middle_loss)
    losses_3.append(loss)

losses_4 = []


for i in range(len(lines4)/12):

    epoch = lines4[i*12:i*12+12]
    first_iter = epoch[0:4]
    second_iter = epoch[4:8]
    last_iter = epoch[8:]

    first_loss = float(first_iter[0].split(',')[2].split(':')[1])
    middle_loss = float(second_iter[0].split(',')[2].split(':')[1])


    first_line = last_iter[0]
    first_line = first_line.split(',')
    first_line = [i.split(':') for i in first_line[2:]]
    loss = float(first_line[0][1])



    # losses.append(first_loss)
    # losses.append(middle_loss)
    losses_4.append(loss)

losses_5 = []


for i in range(len(lines5)/12):

    epoch = lines5[i*12:i*12+12]
    first_iter = epoch[0:4]
    second_iter = epoch[4:8]
    last_iter = epoch[8:]

    first_loss = float(first_iter[0].split(',')[2].split(':')[1])
    middle_loss = float(second_iter[0].split(',')[2].split(':')[1])


    first_line = last_iter[0]
    first_line = first_line.split(',')
    first_line = [i.split(':') for i in first_line[2:]]
    loss = float(first_line[0][1])



    # losses.append(first_loss)
    # losses.append(middle_loss)
    losses_5.append(loss)

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b
# a,b = best_fit(range(len(losses)),losses)
#
# par = np.polyfit(range(len(losses)), losses, 1, full=True)
# print(par)
#
# mode_label = None
# if mode == 'train':
#     mode_label = "Training"
# else:
#     mode_label = "Testing"
#
# fig = plt.figure()
# plt.scatter(range(len(losses)),losses)
#
#
# xd = range(len(losses))
# yd = losses
#
# slope=par[0][0]
# intercept=par[0][1]
# xl = [min(xd), max(xd)]
# yl = [slope*xx + intercept  for xx in xl]



# plt.plot(range(len(losses[:20])), losses[:20], '-r', label = 'c3d attn')
plt.plot(range(len(losses_2[:28])), losses_2[:28], '-b', label = 'c3d')
# plt.plot(range(len(losses_3)[:20]), losses_3[:20], '-g', label = 's3d')
# plt.plot(range(len(losses_4[:20])), losses_4[:20], '-k', label = 's3d attn')
plt.plot(range(len(losses_5[:28])), losses_5[:28], '-m', label = '8 gru attn')
plt.legend()



plt.title(experiment_name, fontsize=20)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('{} Loss'.format(mode_label), fontsize=16)
plt.savefig("Training Losses")