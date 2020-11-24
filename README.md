
# MTL-AQA Architecture Experiments

## MTL-AQA Concept:

<p align="center"> <img src="diving_sample.gif?raw=true" alt="diving_video" width="200"/> </p>
<p align="center"> <img src="mtlaqa_concept.png?raw=true" alt="mtl_net" width="400"/> </p>


Experiments by [Akhila Ballari](https://github.com/aballari9),[Jas Pyneni](https://github.com/jpyneni3), and [Nitya Tarakad](https://github.com/nitarakad)

This project aims to experiment on the MTL-AQA architecture designed by Paritosh Parmar and Brendan Tran Morris, [What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment](https://arxiv.org/abs/1904.04346). This repository is forked from [their code],(https://github.com/ParitoshParmar/MTL-AQA)



### Table of Contents
**[1) Data](#1-data-collection)**<br>
<!-- **[2) Data Pre-Processing/Cleaning](#2-data-pre-processingcleaning)**<br>
**[3) Data Augmentation](#3-data-augmentation)**<br>
**[4) Models](#4-models)**<br>
**[5) Experiment Outputs](#5-experiment-outputs)**<br>
**[6) Web App](#6-web-app)**<br> -->


## 1) Data
To collect the necessary data for this project (about 60 GB), follow these steps:

1)Navigate to /MTL-AQA_dataset_release
2)Run the youtube_mp4_converter.py script that was written by us. This will download the videos of Diving competitions from the links in the Video_List.xlsx file.
3) Run the frame_extractor.sh script to convert the videos into the frames that will be indexed by the dataloader for the project
