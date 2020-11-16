file_name = "train_logging_file_1.txt"
experiment_name = "C3DAVG w/ SGD Backbone"
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
    last_iter = epoch[8:]

    first_line = last_iter[0]
    first_line = first_line.split(',')
    first_line = [i.split(':') for i in first_line[2:]]
    loss = float(first_line[0][1])
    fs_loss = float(first_line[1][1])

    cls_loss = float(last_iter[-3].split(':')[1])
    cap_loss = float(last_iter[-2].split(':')[1])

    losses.append(loss)
    fs_losses.append(fs_loss)
    cls_losses.append(cls_loss)
    cap_losses.append(cap_loss)

print(len(losses))
print(len(fs_losses))
print(len(cls_losses))
print(len(cap_losses))

print()

print(losses[49])
print(fs_losses[49])
print(cls_losses[49])
print(cap_losses[49])

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

import matplotlib.pyplot as plt

mode_label = None
if mode == 'train':
    mode_label = "Training"
else:
    mode_label = "Testing"

fig = plt.figure()
plt.scatter(range(len(losses)),losses)
yfit = [a + b * xi for xi in range(len(losses))]
plt.plot(range(len(losses)), yfit)
fig.suptitle(experiment_name, fontsize=20)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('{} Loss'.format(mode_label), fontsize=16)
plt.show()