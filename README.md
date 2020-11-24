
# MTL-AQA Architecture Experiments

## MTL-AQA Concept:

<p align="center"> <img src="diving_sample.gif?raw=true" alt="diving_video" width="200"/> </p>
<p align="center"> <img src="mtlaqa_concept.png?raw=true" alt="mtl_net" width="400"/> </p>


Experiments by [Akhila Ballari](https://github.com/aballari9),[Jas Pyneni](https://github.com/jpyneni3), and [Nitya Tarakad](https://github.com/nitarakad)

This project aims to experiment on the MTL-AQA architecture designed by Paritosh Parmar and Brendan Tran Morris, [What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment](https://arxiv.org/abs/1904.04346). This repository is forked from [their code],(https://github.com/ParitoshParmar/MTL-AQA)



### Table of Contents
**[1) Data](#1-data-collection)**<br>
**[2) Experiments](#2-experiments)**<br>
<!-- **[3) Data Augmentation](#3-data-augmentation)**<br>
**[4) Models](#4-models)**<br>
**[5) Experiment Outputs](#5-experiment-outputs)**<br>
**[6) Web App](#6-web-app)**<br> -->


## 1) Data
To collect the necessary data for this project (about 60 GB), follow these steps:

1)Navigate to /MTL-AQA_dataset_release  
2)Run the youtube_mp4_converter.py script that was written by us. This will download the videos of Diving competitions from the links in the Video_List.xlsx file.  
3) Run the frame_extractor.sh script to convert the videos into the frames that will be indexed by the dataloader for the project  



## 2) Experiments
The below table, for each experiment, lists which file in /MTL-AQA_code_release to run, necessary changes to make in that file, necessary changes to make in opts.py,  and the location of train and test logs we got for that experiment. For each model, make sure to make a new saving directory and update the saving_dir variable with in the *_test_train.py file corresponding to that experiment.

|  Experiment Number |  Description | Train  |  Test |  Model Folder |
|---|---|---|---| ---|
| 1 | C3DAVG model (baseline)  |  c3davg_train_logging_file_1 | c3davg_test_logging_file_1  | c3davg_140_saved |
| 2 | C3DAVG model with SGD Backbone  |  train_logging_file_1 | test_logging_file_1  |c3davg_140_saved_s3d |  
| 3 | C3DAVG model with Attention  | c3d_attn_train_logging_file_1.txt   | c3d_attn_test_logging_file_1.txt  | c3davg_140_saved_attn  |
|4  |   |   |   | |
