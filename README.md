
# MTL-AQA Architecture Experiments

## MTL-AQA Concept:

<p align="center"> <img src="diving_sample.gif?raw=true" alt="diving_video" width="200"/> </p>
<p align="center"> <img src="mtlaqa_concept.png?raw=true" alt="mtl_net" width="400"/> </p>


Experiments by [Akhila Ballari](https://github.com/aballari9),[Jas Pyneni](https://github.com/jpyneni3), and [Nitya Tarakad](https://github.com/nitarakad)

[View our research paper](research_paper.pdf)

This project aims to experiment on the MTL-AQA architecture designed by Paritosh Parmar and Brendan Tran Morris, [What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment](https://arxiv.org/abs/1904.04346). This repository is forked from [their code],(https://github.com/ParitoshParmar/MTL-AQA)



### Table of Contents
**[1) Data](#1-data-collection)**<br>
**[2) Setup](#2-setup)**<br>
**[3) Experiments](#3-experiments)**<br>
**[4) Analysis](#4-analysis)**<br>

### Requirements

```
Pytorch
ffmpeg
pytube
numpy
scipy
```
## 1) Data
To collect the necessary data for this project (about 60 GB), follow these steps:

1)Navigate to /MTL-AQA_dataset_release  
2)Run the youtube_mp4_converter.py script that was written by us. This will download the videos of Diving competitions from the links in the Video_List.xlsx file.  
3) Run the frame_extractor.sh script to convert the videos into the frames that will be indexed by the dataloader for the project  

## 2) Setup
Follow the directions in /MTL-AQA_code_release/readme.md to download the pre-trained weights for both backbones   

For models that use attention for the downstream task of generating captions, set the use_attn=True variable, in the corresponding train_test.py file for that model, in the initialization of the S2VTModel.  

Make the necessary variable chances in opts.py based upon the table below  

## 3) Experiments
The below table, for each experiment, lists which file in /MTL-AQA_code_release to run, necessary variable changes to make in opts.py,  and the location of train and test logs we got for that experiment. For each model, make sure to make a new saving directory and update the saving_dir variable with in the *_test_train.py file corresponding to that experiment.

|  Experiment Number |  Description | File (run) | Changes | Train Log |   Test Log |
|---|---|---|---| ---| ---|
| 0 | C3DAVG model (baseline)  | train_test_C3DAVG.py | caption_lstm_cell_type = 'gru'   caption_lstm_num_layers = 2 |c3davg_train_logging_file_1.txt | c3davg_test_logging_file_1.txt  |
| 1 | C3DAVG model with Attention  |  train_test_C3DAVG_S3D.py | use_attn=True caption_lstm_cell_type = 'gru'   caption_lstm_num_layers = 2 | c3d_attn_train_logging_file_1.txt | c3d_attn_test_logging_file_1.txt  |
| 2 | C3DAVG model with SGD Backbone  |  train_test_C3DAVG_S3D.py | caption_lstm_cell_type = 'gru'   caption_lstm_num_layers = 2 | train_logging_file_1.txt | test_logging_file_1.txt  |
| 3 | C3DAVG model with SGD Backbone and Attention  | train_test_C3DAVG_S3D.py | use_attn=True caption_lstm_cell_type = 'gru'   caption_lstm_num_layers = 2 | s3d_attn_train_logging_file_1.txt   | s3d_attn_test_logging_file_1.txt  |
| 4 | C3DAVG model with Attention and stacked GRUs |  train_test_C3DAVG.py | use_attn=True caption_lstm_cell_type = 'gru'   caption_lstm_num_layers = 8 | c3davg_8_gru_attn_train_logging_file_1.txt | c3davg_8_gru_attn_test_logging_file_1.txt  |
| 5 | C3DAVG model with Attention and stacked LSTMS |  train_test_C3DAVG.py | use_attn=True caption_lstm_cell_type = 'lstm'   caption_lstm_num_layers = 8 | c3davg_8_lstm_attn_train_logging_file_1.txt | c3davg_8_lstm_attn_test_logging_file_1.txt  |
| 6 | C3DAVG model with 2 LSTMs |  train_test_C3DAVG.py | caption_lstm_cell_type = 'LSTM'   caption_lstm_num_layers = 2 |  | |
| 7 | C3DAVG model with LSTM encoding of frames |  train_test_LSTM_autoencoder.py | caption_lstm_cell_type = 'gru'   caption_lstm_num_layers = 2 |  | |

## 4) Analysis
Once the log files are populated, run ```parse.py``` and ```parse_test.py``` to get the graphs from our paper
