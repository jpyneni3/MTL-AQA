# Author: Paritosh Parmar (https://github.com/ParitoshParmar)
# Code used in the following, also if you find it useful, please consider citing the following:
#
# @inproceedings{parmar2019and,
#   title={What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment},
#   author={Parmar, Paritosh and Tran Morris, Brendan},
#   booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
#   pages={304--313},
#   year={2019}
# }

import os
import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_C3DAVG import VideoDataset
import random
import scipy.stats as stats
import torch.optim as optim
import torch.nn as nn
from models.C3DAVG.C3D_altered import C3D_altered
from models.C3DAVG.S3D_model import S3D
from models.C3DAVG.LSTM_autoencoder import AutoEncoderRNN # for lstm
from models.C3DAVG.LSTM_autoencoder import EncoderRNN # for lstm
from models.C3DAVG.img_to_vec import Img2Vec # for lstm
from models.C3DAVG.my_fc6 import my_fc6
from models.C3DAVG.score_regressor import score_regressor
from models.C3DAVG.dive_classifier import dive_classifier
from models.C3DAVG.S2VTModel import S2VTModel
from opts import *
from utils import utils_1
import numpy as np

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True

current_run = 1 # CHANGE THIS FOR LOG FILE (BEFORE RUNNING HAPPENS)
# train_logging_file_name = "c3davg_train_logging_file_" + str(current_run) + ".txt"
train_logging_file_name = "c3davg_lstm_encoder_train_logging_file_" + str(current_run) + ".txt"
train_logging_file = open(train_logging_file_name,"x")
train_logging_file.close()
# test_logging_file_name = "c3davg_test_logging_file_" + str(current_run) + ".txt"
test_logging_file_name = "c3davg_lstm_encoder_test_logging_file_" + str(current_run) + ".txt"
test_logging_file = open(test_logging_file_name, "x")
test_logging_file.close()

extractor = Img2Vec() # use extractor.get_vec(img) with img being the frame


def save_model(model, model_name, epoch, path):
    model_path = os.path.join(path, '%s_%d.pth' % (model_name, epoch))
    torch.save(model.state_dict(), model_path)


def train_phase(train_dataloader, optimizer, criterions, epoch):
    criterion_final_score = criterions['criterion_final_score']; penalty_final_score = criterions['penalty_final_score']
    if with_dive_classification:
        criterion_dive_classifier = criterions['criterion_dive_classifier']
    if with_caption:
        criterion_caption = criterions['criterion_caption']

    model_CNN.train()
    model_my_fc6.train()
    model_score_regressor.train()
    if with_dive_classification:
        model_dive_classifier.train()
    if with_caption:
        model_caption.train()

    iteration = 0
    for data in train_dataloader:
        true_final_score = data['label_final_score'].unsqueeze_(1).type(torch.FloatTensor).cuda()
        if with_dive_classification:
            true_postion = data['label_position'].cuda()
            true_armstand = data['label_armstand'].cuda()
            true_rot_type = data['label_rot_type'].cuda()
            true_ss_no = data['label_ss_no'].cuda()
            true_tw_no = data['label_tw_no'].cuda()
        if with_caption:
            true_captions = data['label_captions'].cuda()
            true_captions_mask = data['label_captions_mask'].cuda()
        video = data['video'].transpose_(1, 2).cuda()

        batch_size, C, frames, H, W = video.shape
        clip_feats = torch.Tensor([]).cuda()
        print("video shape")
        print(video.shape)

        for i in np.arange(0, frames - 17, 16):
            clip = video[:, :, i:i + 16, :, :]
            #clip = extractor.get_vec(clip)
            #print("shape of clip before cnn")
            #print(clip.shape)
            clip_feats_temp = model_CNN(clip)
            #print("shape after cnn")
            #print(clip_feats_temp.shape)
            clip_feats_temp.unsqueeze_(0)
            clip_feats_temp.transpose_(0, 1)
            #print("shape after unsqueeze and transpose")
            #print(clip_feats_temp.shape)
            clip_feats = torch.cat((clip_feats, clip_feats_temp), 1)
            #print("final shape in the loop")
            #print(clip_feats.shape)
        #print("CLIP FEAT AVG TRAIN SHAPE BEFORE: ", clip_feats.shape)
        #clip_feats_avg = clip_feats.mean(1)
        #print("CLIP FEAT AVG TRAIN SHAPE AFTER: ", clip_feats_avg.shape)

        clip_feats_lstm = lstm_feature_encoder(clip_feats)
        #sample_feats_fc6 = model_my_fc6(clip_feats_avg)
        sample_feats_fc6 = model_my_fc6(clip_feats_lstm)
        pred_final_score = model_score_regressor(sample_feats_fc6)
        if with_dive_classification:
            (pred_position, pred_armstand, pred_rot_type, pred_ss_no,
             pred_tw_no) = model_dive_classifier(sample_feats_fc6)
        if with_caption:
            seq_probs, _ = model_caption(clip_feats, true_captions, 'train')

        loss_final_score = (criterion_final_score(pred_final_score, true_final_score)
                            + penalty_final_score(pred_final_score, true_final_score))
        loss = 0
        loss += loss_final_score
        if with_dive_classification:
            loss_position = criterion_dive_classifier(pred_position, true_postion)
            loss_armstand = criterion_dive_classifier(pred_armstand, true_armstand)
            loss_rot_type = criterion_dive_classifier(pred_rot_type, true_rot_type)
            loss_ss_no = criterion_dive_classifier(pred_ss_no, true_ss_no)
            loss_tw_no = criterion_dive_classifier(pred_tw_no, true_tw_no)
            loss_cls = loss_position + loss_armstand + loss_rot_type + loss_ss_no + loss_tw_no
            loss += loss_cls
        if with_caption:
            loss_caption = criterion_caption(seq_probs, true_captions[:, 1:], true_captions_mask[:, 1:])
            loss += loss_caption*0.01

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 20 == 0:
            train_logging_file = open(train_logging_file_name, "a")
            train_output = f"Epoch: {epoch}, Iter: {iteration}, Loss: {loss}, FS Loss: {loss_final_score}"
            print(train_output, end="")
            train_logging_file.write(train_output + "\n")
            if with_dive_classification:
                  print(' Cls Loss: ', loss_cls, end="")
                  train_logging_file.write(f"Cls Loss: {loss_cls}" + "\n")
            if with_caption:
                  print(' Cap Loss: ', loss_caption, end="")
                  train_logging_file.write(f"Cap Loss: {loss_caption}" + "\n")
            print(' ')
            train_logging_file.write("\n")
            train_logging_file.close()
        iteration += 1


def test_phase(test_dataloader):
    print('In testphase...')
    with torch.no_grad():
        pred_scores = []; true_scores = []
        if with_dive_classification:
            pred_position = []; pred_armstand = []; pred_rot_type = []; pred_ss_no = []; pred_tw_no = []
            true_position = []; true_armstand = []; true_rot_type = []; true_ss_no = []; true_tw_no = []

        model_CNN.eval()
        model_my_fc6.eval()
        model_score_regressor.eval()
        if with_dive_classification:
            model_dive_classifier.eval()
        if with_caption:
            model_caption.eval()

        for data in test_dataloader:
            true_scores.extend(data['label_final_score'].data.numpy())
            if with_dive_classification:
                true_position.extend(data['label_position'].numpy())
                true_armstand.extend(data['label_armstand'].numpy())
                true_rot_type.extend(data['label_rot_type'].numpy())
                true_ss_no.extend(data['label_ss_no'].numpy())
                true_tw_no.extend(data['label_tw_no'].numpy())
            video = data['video'].transpose_(1, 2).cuda()

            batch_size, C, frames, H, W = video.shape
            clip_feats = torch.Tensor([]).cuda()

            for i in np.arange(0, frames - 17, 16):
                clip = video[:, :, i:i + 16, :, :]
                #clip = extractor.get_vec(clip)
                clip_feats_temp = model_CNN(clip)
                clip_feats_temp.unsqueeze_(0)
                clip_feats_temp.transpose_(0, 1)
                clip_feats = torch.cat((clip_feats, clip_feats_temp), 1)
            #print("CLIP FEAT AVG TRAIN SHAPE BEFORE: ", clip_feats.shape)
            #clip_feats_avg = clip_feats.mean(1)
            #print("CLIP FEAT AVG TRAIN SHAPE AFTER: ", clip_feats_avg.shape)
            #clip_feats_lstm = lstm_feature_encoder(clip_feats)
            clip_feats_lstm = lstm_feature_encoder(clip_feats)

            #sample_feats_fc6 = model_my_fc6(clip_feats_avg)
            sample_feats_fc6 = model_my_fc6(clip_feats_lstm)

            temp_final_score = model_score_regressor(sample_feats_fc6)
            pred_scores.extend([element[0] for element in temp_final_score.data.cpu().numpy()])
            if with_dive_classification:
                temp_position, temp_armstand, temp_rot_type, temp_ss_no, temp_tw_no = model_dive_classifier(sample_feats_fc6)
                softmax_layer = nn.Softmax(dim=1)
                temp_position = softmax_layer(temp_position).data.cpu().numpy()
                temp_armstand = softmax_layer(temp_armstand).data.cpu().numpy()
                temp_rot_type = softmax_layer(temp_rot_type).data.cpu().numpy()
                temp_ss_no = softmax_layer(temp_ss_no).data.cpu().numpy()
                temp_tw_no = softmax_layer(temp_tw_no).data.cpu().numpy()

                for i in range(len(temp_position)):
                    pred_position.extend(np.argwhere(temp_position[i] == max(temp_position[i]))[0])
                    pred_armstand.extend(np.argwhere(temp_armstand[i] == max(temp_armstand[i]))[0])
                    pred_rot_type.extend(np.argwhere(temp_rot_type[i] == max(temp_rot_type[i]))[0])
                    pred_ss_no.extend(np.argwhere(temp_ss_no[i] == max(temp_ss_no[i]))[0])
                    pred_tw_no.extend(np.argwhere(temp_tw_no[i] == max(temp_tw_no[i]))[0])

        test_logging_file = open(test_logging_file_name, "a")
        if with_dive_classification:
            position_correct = 0; armstand_correct = 0; rot_type_correct = 0; ss_no_correct = 0; tw_no_correct = 0
            for i in range(len(pred_position)):
                if pred_position[i] == true_position[i]:
                    position_correct += 1
                if pred_armstand[i] == true_armstand[i]:
                    armstand_correct += 1
                if pred_rot_type[i] == true_rot_type[i]:
                    rot_type_correct += 1
                if pred_ss_no[i] == true_ss_no[i]:
                    ss_no_correct += 1
                if pred_tw_no[i] == true_tw_no[i]:
                    tw_no_correct += 1
            position_accu = position_correct / len(pred_position) * 100
            armstand_accu = armstand_correct / len(pred_armstand) * 100
            rot_type_accu = rot_type_correct / len(pred_rot_type) * 100
            ss_no_accu = ss_no_correct / len(pred_ss_no) * 100
            tw_no_accu = tw_no_correct / len(pred_tw_no) * 100
            print('Accuracies: Position: ', position_accu, ' Armstand: ', armstand_accu, ' Rot_type: ', rot_type_accu,
                  ' SS_no: ', ss_no_accu, ' TW_no: ', tw_no_accu)
            test_logging_file.write(f"Position: {position_accu}, Armstand: {armstand_accu}, Rot_type: {rot_type_accu}, SS_no: {ss_no_accu}, TW_no: {tw_no_accu}" + "\n")

        rho, p = stats.spearmanr(pred_scores, true_scores)
        print('Predicted scores: ', pred_scores)
        print('True scores: ', true_scores)
        print('Correlation: ', rho)
        test_logging_file.write(f"Correlation: {rho}" + "\n")
        test_logging_file.write("\n")
        test_logging_file.close()


def main():
    parameters_2_optimize = (list(model_CNN.parameters()) + list(model_my_fc6.parameters()) +
                           list(model_score_regressor.parameters()))
    parameters_2_optimize_named = (list(model_CNN.named_parameters()) + list(model_my_fc6.named_parameters()) +
                                   list(model_score_regressor.named_parameters()))
    if with_dive_classification:
        parameters_2_optimize = parameters_2_optimize + list(model_dive_classifier.parameters())
        parameters_2_optimize_named = parameters_2_optimize_named + list(model_dive_classifier.named_parameters())
    if with_caption:
        parameters_2_optimize = parameters_2_optimize + list(model_caption.parameters())
        parameters_2_optimize_named = parameters_2_optimize_named + list(model_caption.named_parameters())

    optimizer = optim.Adam(parameters_2_optimize, lr=0.0001)
    # print('Parameters that will be learnt: ', parameters_2_optimize_named)

    criterions = {}
    criterion_final_score = nn.MSELoss()
    penalty_final_score = nn.L1Loss()
    criterions['criterion_final_score'] = criterion_final_score
    criterions['penalty_final_score'] = penalty_final_score
    if with_dive_classification:
        criterion_dive_classifier = nn.CrossEntropyLoss()
        criterions['criterion_dive_classifier'] = criterion_dive_classifier
    if with_caption:
        criterion_caption = utils_1.LanguageModelCriterion()
        criterions['criterion_caption'] = criterion_caption

    train_dataset = VideoDataset('train')
    test_dataset = VideoDataset('test')
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    print('Length of train loader: ', len(train_dataloader))
    print('Length of test loader: ', len(test_dataloader))
    print('Training set size: ', len(train_dataloader)*train_batch_size,
          ';    Test set size: ', len(test_dataloader)*test_batch_size)

    # actual training, testing loops
    for epoch in range(100):
        # saving_dir = 'c3davg_140_saved' # ADDED PATH FOR SAVING DIRECTORY
        saving_dir = 'c3davg_lstm_encode' # ADDED PATH FOR SAVING DIRECTORY
        print('-------------------------------------------------------------------------------------------------------')
        for param_group in optimizer.param_groups:
            print('Current learning rate: ', param_group['lr'])

        train_phase(train_dataloader, optimizer, criterions, epoch)
        test_phase(test_dataloader)

        if (epoch+1) % model_ckpt_interval == 0: # save models every 5 epochs
            save_model(model_CNN, 'model_CNN', epoch, saving_dir)
            save_model(model_my_fc6, 'model_my_fc6', epoch, saving_dir)
            save_model(model_score_regressor, 'model_score_regressor', epoch, saving_dir)
            if with_dive_classification:
                save_model(model_dive_classifier, 'model_dive_classifier', epoch, saving_dir)
            if with_caption:
                save_model(model_caption, 'model_caption', epoch, saving_dir)



if __name__ == '__main__':
    # loading the altered C3D backbone (ie C3D upto before fc-6)
    model_CNN_pretrained_dict = torch.load('c3d.pickle')
    #model_CNN_pretrained_dict = torch.load('S3D_kinetics400.pt')
    model_CNN = C3D_altered()
    #print("using s3d")
    #model_CNN = S3D(num_classes)
    print(model_CNN)
    model_CNN_dict = model_CNN.state_dict()
    model_CNN_pretrained_dict = {k: v for k, v in model_CNN_pretrained_dict.items() if k in model_CNN_dict}
    model_CNN_dict.update(model_CNN_pretrained_dict)
    model_CNN.load_state_dict(model_CNN_dict)
    model_CNN = model_CNN.cuda()

    # LSTM encoder for feature extraction
    # clip_feats shape  = [3,6,8192]
    input_size = 3
    hidden_size = 8192
    num_layers = 2

    lstm_feature_encoder = EncoderRNN(input_size, hidden_size, num_layers, False)
    lstm_feature_encoder.cuda()

    # loading our fc6 layer
    model_my_fc6 = my_fc6()
    model_my_fc6.cuda()

    # loading our score regressor
    model_score_regressor = score_regressor()
    model_score_regressor = model_score_regressor.cuda()
    print('Using Final Score Loss')

    if with_dive_classification:
        # loading our dive classifier
        model_dive_classifier = dive_classifier()
        model_dive_classifier = model_dive_classifier.cuda()
        print('Using Dive Classification Loss')

    if with_caption:
        # loading our caption model
        model_caption = S2VTModel(vocab_size, max_cap_len, caption_lstm_dim_hidden,
                                  caption_lstm_dim_word, caption_lstm_dim_vid,
                                  rnn_cell=caption_lstm_cell_type, n_layers=caption_lstm_num_layers,
                                  rnn_dropout_p=caption_lstm_dropout,
                                  use_attention=False)
        model_caption = model_caption.cuda()
        print('Using Captioning Loss')

    main()
