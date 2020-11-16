import numpy as np
import matplotlib.pyplot as plt

file_name = "c3d_attn_train_logging_file_1.txt"
experiment_name = "C3DAVG w/ Attenion"
mode = "train" #or test
f = open(file_name, 'r')
lines = f.readlines()

num_epochs = len(lines)/12
print(num_epochs)

losses = []
fs_losses = []
cls_losses = []
cap_losses = []

for i in range(num_epochs):

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

print(len(losses))
print(len(fs_losses))
print(len(cls_losses))
print(len(cap_losses))

print()



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
a,b = best_fit(range(len(losses)),losses)

par = np.polyfit(range(len(losses)), losses, 1, full=True)
print(par)

mode_label = None
if mode == 'train':
    mode_label = "Training"
else:
    mode_label = "Testing"

fig = plt.figure()
plt.scatter(range(len(losses)),losses)


xd = range(len(losses))
yd = losses

slope=par[0][0]
intercept=par[0][1]
xl = [min(xd), max(xd)]
yl = [slope*xx + intercept  for xx in xl]



plt.plot(xl, yl, '-r')



fig.suptitle(experiment_name, fontsize=20)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('{} Loss'.format(mode_label), fontsize=16)
plt.show()