import numpy as np
import matplotlib.pyplot as plt

dir = "logs/"
saving_dir = "loss_results/"
file_name = "c3davg_train_logging_file_1.txt"
file_name2 = "c3d_attn_train_logging_file_1.txt"
# file_name = "c3d_attn_train_logging_file_1.txt"
# file_name2 = "c3davg_train_logging_file_1.txt"
file_name3 = "train_logging_file_1.txt"
file_name4 = "s3d_attn_train_logging_file_1.txt"
file_name5 = "c3davg_8_gru_attn_train_logging_file_1.txt"
file_name6 = "c3davg_8_lstm_attn_train_logging_file_1.txt"
file_name7 = "c3davg_8_gru_train_logging_file_1.txt"

experiment_name = "Training Losses"

mode = "train" #or test

mode_label = None
if mode == 'train':
    mode_label = "Training"
else:
    mode_label = "Testing"

def parse_logs(file_name):
    f = open(dir + file_name, 'r')
    lines = f.readlines()
    losses = []

    for i in range(int(len(lines)/12)):

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

        losses.append(loss)
    return losses

if __name__ == "__main__":
    losses = parse_logs(file_name)

    losses_2 = parse_logs(file_name2)

    losses_3 = parse_logs(file_name3)

    losses_4 = parse_logs(file_name4)

    losses_5 = parse_logs(file_name5)

    losses_6 = parse_logs(file_name6)

    # losses_7 = parse_logs(file_name7)
    
    plt.figure(0)
    plt.plot(range(len(losses)), losses, '-b', linewidth=1, label = 'C3D')
    plt.plot(range(len(losses_2)), losses_2, '-r', linewidth=1, label = 'C3D + Attn')
    plt.plot(range(len(losses_3)), losses_3, '-g', linewidth=1, label = 'S3D')
    plt.plot(range(len(losses_4)), losses_4, '-k', linewidth=1, label = 'S3D + Attn')
    plt.plot(range(len(losses_5)), losses_5, '-m', linewidth=1, label = 'C3D + 8 GRUs + Attn')
    plt.plot(range(len(losses_6)), losses_6, '-c', linewidth=1, label = 'C3D + 8 LSTMs + Attn')
    # plt.plot(range(len(losses_7)), losses_7, 'y', linewidth=1, label = "C3D + 8 GRUs")
    plt.legend(loc = 'best')

    plt.title(experiment_name, fontsize=20)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('{} Loss'.format(mode_label), fontsize=16)
    plt.savefig(saving_dir + "Training Losses")

    # 1:1 Comparisons
    plt.figure(1)
    plt.title('(a) C3D vs C3D + Attn')
    plt.plot(range(len(losses)), losses, '-b', linewidth=1, label = 'C3D')
    plt.plot(range(len(losses_2)), losses_2, '-r', linewidth=1, label = 'C3D + Attn')
    plt.legend(loc = 'best')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('{} Loss'.format(mode_label), fontsize=16)
    plt.savefig(saving_dir + "C3D vs C3D + Attn")

    plt.figure(2)
    plt.title('(b) C3D vs S3D')
    plt.plot(range(len(losses)), losses, '-b', linewidth=1, label = 'C3D')
    plt.plot(range(len(losses_3)), losses_3, '-g', linewidth=1, label = 'S3D')
    plt.legend(loc = 'best')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('{} Loss'.format(mode_label), fontsize=16)
    plt.savefig(saving_dir + "C3D vs S3D")

    plt.figure(3)
    plt.title('(c) C3D vs S3D + Attn')
    plt.plot(range(len(losses)), losses, '-b', linewidth=1, label = 'C3D')
    plt.plot(range(len(losses_4)), losses_4, '-k', linewidth=1, label = 'S3D + Attn')
    plt.legend(loc = 'best')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('{} Loss'.format(mode_label), fontsize=16)
    plt.savefig(saving_dir + "C3D vs S3D + Attn")

    plt.figure(4)
    plt.title('(d) C3D vs C3D + 8 GRUS + Attn')
    plt.plot(range(len(losses)), losses, '-b', linewidth=1, label = 'C3D')
    plt.plot(range(len(losses_5)), losses_5, '-m', linewidth=1, label = 'C3D + 8 GRUs + Attn')
    plt.legend(loc = 'best')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('{} Loss'.format(mode_label), fontsize=16)
    plt.savefig(saving_dir + "C3D vs C3D + 8 GRUS + Attn")

    plt.figure(5)
    plt.title('(e) C3D vs C3D + 8 LSTMs + Attn')
    plt.plot(range(len(losses)), losses, '-b', linewidth=1, label = 'C3D')
    plt.plot(range(len(losses_6)), losses_6, '-c', linewidth=1, label = 'C3D + 8 LSTMs + Attn')
    plt.legend(loc = 'best')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('{} Loss'.format(mode_label), fontsize=16)
    plt.savefig(saving_dir + "C3D vs C3D + 8 LSTMs + Attn")

    # Bar Graph
    plt.figure(6)
    plt.title('Final Losses')
    x = ['C3D', 'C3D + Attn', 'S3D', 'S3D + Attn', 'C3D + 8 GRUs + Attn', 'C3D + 8 LSTMs + Attn']
    avgs = [losses[-1], losses_2[-1], losses_3[-1], losses_4[-1], losses_5[-1], losses_6[-1]]

    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, avgs, color='green')
    plt.xlabel('Experiment', fontsize=18)
    plt.ylabel('{} Final Loss'.format(mode_label), fontsize=16)
    plt.xticks(x_pos, x)
    plt.savefig(saving_dir + "bar")
