import numpy as np
import matplotlib.pyplot as plt

dir = "logs/"
saving_dir = "correlation_results/"
file_name = "c3davg_test_logging_file_1.txt"
file_name2 = "c3d_attn_test_logging_file_1.txt"
# file_name = "c3d_attn_test_logging_file_1.txt"
# file_name2 = "c3davg_test_logging_file_1.txt"
file_name3 = "test_logging_file_1.txt"
file_name4 = "s3d_attn_test_logging_file_1.txt"
file_name5 = "c3davg_8_gru_attn_test_logging_file_1.txt"
file_name6 = "c3davg_8_lstm_attn_test_logging_file_1.txt"

experiment_name = "Testing Correlations"

mode = "test" #or test

mode_label = None
if mode == 'train':
    mode_label = "Training"
else:
    mode_label = "Testing"

def parse_logs(file_name):
    f = open(dir + file_name, 'r')
    lines = f.readlines()

    corr_list = []
    for i in range(int(len(lines)/3)):
        epoch = lines[i*3:i*3+3]
        corr = float(epoch[1:2][0].split(':')[1])
        corr_list.append(corr)

    return corr_list

if __name__ == "__main__":
    corr1 = parse_logs(file_name)

    corr2 = parse_logs(file_name2)

    corr3 = parse_logs(file_name3)

    corr4 = parse_logs(file_name4)

    corr5 = parse_logs(file_name5)

    corr6 = parse_logs(file_name6)

    plt.figure(0)
    plt.plot(range(len(corr1)), corr1, '-b', label = 'C3D')
    plt.plot(range(len(corr2)), corr2, '-r', label = 'C3D + Attn')
    plt.plot(range(len(corr3)), corr3, '-g', label = 'S3D')
    plt.plot(range(len(corr4)), corr4, 'orange', label = 'S3D + Attn')
    plt.plot(range(len(corr5)), corr5, '-m', label = 'C3D + 8 GRUs + Attn')
    plt.plot(range(len(corr6)), corr6, '-c', label = 'C3D + 8 LSTMs + Attn')
    plt.legend(loc = 'best')


    plt.title(experiment_name, fontsize=20)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Correlation'.format(mode_label), fontsize=16)
    plt.savefig(saving_dir + "Testing_Correlations")

    # 1:1 Comparisons
    plt.figure(1)
    plt.title('C3D vs C3D + Attn', fontsize=20)
    plt.plot(range(len(corr1)), corr1, '-b', linewidth=1, label = 'C3D')
    plt.plot(range(len(corr2)), corr2, '-r', linewidth=1, label = 'C3D + Attn')
    plt.legend(loc = 'best')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('{} Correlation'.format(mode_label), fontsize=16)
    plt.savefig(saving_dir + "exp_1_corr")

    plt.figure(2)
    plt.title('C3D vs S3D', fontsize=20)
    plt.plot(range(len(corr1)), corr1, '-b', linewidth=1, label = 'C3D')
    plt.plot(range(len(corr3)), corr3, '-g', linewidth=1, label = 'S3D')
    plt.legend(loc = 'best')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('{} Correlation'.format(mode_label), fontsize=16)
    plt.savefig(saving_dir + "exp_2_corr")

    plt.figure(3)
    plt.title('C3D vs S3D + Attn', fontsize=20)
    plt.plot(range(len(corr1)), corr1, '-b', linewidth=1, label = 'C3D')
    plt.plot(range(len(corr4)), corr4, 'orange', linewidth=1, label = 'S3D + Attn')
    plt.legend(loc = 'best')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('{} Correlation'.format(mode_label), fontsize=16)
    plt.savefig(saving_dir + "exp_3_corr")

    plt.figure(4)
    plt.title('C3D vs C3D + 8 GRUS + Attn', fontsize=20)
    plt.plot(range(len(corr1)), corr1, '-b', linewidth=1, label = 'C3D')
    plt.plot(range(len(corr5)), corr5, '-m', linewidth=1, label = 'C3D + 8 GRUs + Attn')
    plt.legend(loc = 'best')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('{} Correlation'.format(mode_label), fontsize=16)
    plt.savefig(saving_dir + "exp_4_corr")

    plt.figure(5)
    plt.title('C3D vs C3D + 8 LSTMs + Attn', fontsize=20)
    plt.plot(range(len(corr1)), corr1, '-b', linewidth=1, label = 'C3D')
    plt.plot(range(len(corr6)), corr6, '-c', linewidth=1, label = 'C3D + 8 LSTMs + Attn')
    plt.legend(loc = 'best')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('{} Correlation'.format(mode_label), fontsize=16)
    plt.savefig(saving_dir + "exp_5_corr")

    # Bar Graph
    plt.figure(6)
    plt.title('Final Correlations', fontsize=20)
    x = ['C3D', 'C3D + Attn', 'S3D', 'S3D + Attn', 'C3D + 8 GRUs + Attn', 'C3D + 8 LSTMs + Attn']
    avgs = [corr1[-1], corr2[-1], corr3[-1], corr4[-1], corr5[-1], corr6[-1]]

    x_pos = [i for i, _ in enumerate(x)]

    barlist = plt.bar(x_pos, avgs, color='green')
    barlist[0].set_color('b')
    barlist[1].set_color('r')
    barlist[2].set_color('g')
    barlist[3].set_color('orange')
    barlist[4].set_color('m')
    barlist[5].set_color('c')

    plt.xlabel('Experiment', fontsize=18)
    plt.ylabel('{} Final Correlations'.format(mode_label), fontsize=16)
    plt.xticks(x_pos, x, rotation=15)
    plt.tight_layout()
    plt.savefig(saving_dir + "bar_corr")

    # Baseline
    plt.figure(7)
    plt.title('C3D Baseline', fontsize=20)
    plt.plot(range(len(corr1)), corr1, '-b', linewidth=1, label = 'C3D')
    plt.legend(loc = 'best')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('{} Correlation'.format(mode_label), fontsize=16)
    plt.savefig(saving_dir + "baseline_corr")
