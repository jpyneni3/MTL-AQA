
# MTL-AQA Architecture Experiments

## MTL-AQA Concept:

<p align="center"> <img src="diving_sample.gif?raw=true" alt="diving_video" width="200"/> </p>
<p align="center"> <img src="mtlaqa_concept.png?raw=true" alt="mtl_net" width="400"/> </p>


Experiments by [Akhila Ballari](https://github.com/aballari9),[Jas Pyneni](https://github.com/jpyneni3), and [Nitya Tarakad](https://github.com/nitarakad)

This project aims to experiment on the MTL-AQA architecture designed by Paritosh Parmar and Brendan Tran Morris, [What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment](https://arxiv.org/abs/1904.04346). This repository is forked from [their code],(https://github.com/ParitoshParmar/MTL-AQA)



### Table of Contents
**[1) Data Collection](#1-data-collection)**<br>
**[2) Data Pre-Processing/Cleaning](#2-data-pre-processingcleaning)**<br>
**[3) Data Augmentation](#3-data-augmentation)**<br>
**[4) Models](#4-models)**<br>
**[5) Experiment Outputs](#5-experiment-outputs)**<br>
**[6) Web App](#6-web-app)**<br>



| Directory               | Description                                                              |
|-------------------------|--------------------------------------------------------------------------|
| Augmentation Data | Cleaned data generated through data augmentation |
| Cleaned Data | Aggressively cleaned original data  |
| Experiments | PDF notebook outputs and documentation for each experiment |
| Notebooks | Notebooks used for training classifier models and data aggregation/augmentation |
| Uncleaned Data | original data without cleaning |

## 1) Data Collection
### Requirements (only for Scrapper tool)
Python Libraries:
```
pandas
feedparser
json
newspaper
csv
urllib.parse
```

| File | Location | Source|
|--------------|--------------------------|----------------------|
| LIAR Dataset | Uncleaned Data/liar_dataset | https://www.cs.ucsb.edu/william/data/liar_dataset.zip |
| Kaggle Fake News | Uncleaned Data/Human_Fake_News.csv | https://www.kaggle.com/mrisdal/fake-news |
| Kaggle Real News | Cleaned Data/Kaggle3_Real.csv | https://www.kaggle.com/snapcrack/all-the-news |
| Kaggle Mixed News | Uncleaned Data/fake-news | https://www.kaggle.com/c/fake-news/data |
| Web scraped Real News | Uncleaned Data/Real_News.csv | Scraped using tool|

Most of the the datasets were downloaded directly from the sources. The Kaggle Real News dataset was too large to put on Github before cleaning so it was directly added to cleaned data after pre-processing for necessary articles.

The web scraped real news was obtained by running the scrapper tool (Notebooks/Scrapper.ipynb). The tool reads from specified RSS feeds (Notebooks/News_RSS.json) to extract the latest news. It was ran several times over the course of five days to obtain the current set of data.


## 2) Data Pre-Processing/Cleaning
### Requirements
Python Libraries:
```
nltk
pandas
numpy
```

There are three times where the data needs to be cleaned:
1) Unclean Data  --> Cleaned Data :
    Agressive cleaning of collected data to run base models on
2) Unclean Data --> Semi Clean Data:
    Semi cleaning of collected data to prepare for data augmentation
3) Augmentation Data --> Clean Augmentation Data:
    Agressive cleaning of augmented data to run models on

In each step, the data cleaning notebook is at Notebooks/DataPreprocessing/DataCleaning.ipynb. In the notebook, you can specify where to pull the data from and where to put after cleaning. For either of the aggressive cleanings, use the clean_column method and for the middle process, use the semi_clean_column method. The notebook right now contains the code to agressively clean the Synonym Replacement Augmentation Data for readying for models.

## 3) Data Augmentation
### Requirements:
```
pandas
nltk
torch
fastBPE
sacremoses
```
| Augmentation Technique   | Location  | Steps |
|-------------------------|---------------------|-------------------------------------------|
|Synonym Replacement| Notebooks/DataAugmentation/Synonym Replacement.ipynb | Run the notebook and it will automatically collect the semi clean data and run Synonym Replacement. Takes about 5 minutes.|
|Backtranslation | Notebooks/DataAugmentation/Backtranslation.ipynb | Highly recommend to run on Google Colab with GPU Enabled. Copy the semi cleaned data into the "contents" and give access to your google drive account, from where you can take the backtranslation data. Takes about 26 hours with GPU.|
|Grover Generation | Notebooks/DataAugmentation/GroverGeneration.ipynb | Run the notebook on Google Colab as it requires a GPU to run the Grover model. The notebook automatically downloads the necessary files for the model. You will need to upload the necessary data (LIAR.csv) to the appropriate folder (shown in the notebook) manually. This is probably the most resource intensive task in our project and will take several hours to finish generation.|

Data in the Augmentation Data/semi cleaned is the data upon which to run data augmentation. The outputs of those are placed in the parent directory Augmentation Data with a \_SR or \_BT appended to the file name to denote output of Synonym Replacement or Backtranslation, respectively. Grover augmented data is simply called Grover_clean.csv. Furthermore, this augmented data is pulled from the Augmentation Data for aggressive cleaning (3rd cleaning step above) and placed back in the same directory with a \_clean appended to the file name.
Examples:

a) Augmentation Data/semi cleaned/LIAR.csv --> LIAR dataset semi cleaned to run Data Augmentation on  
b) Augmentation Data/LIAR_SR.csv --> Output of running Synonym Replacement on semi cleaned LIAR dataset  
c) Augmentation Data/LIAR_SR_clean.csv --> Aggressively cleaned Synonym Replacement output from LIAR dataset

## 4) Models
### Requirements:
```
pandas
torch/torchvision
sklearn
keras
tensorflow
transformers (hugging-face)
matplotlib
wordcloud
nltk
```

Three specific models were trained to detect fake news: Naive Bayes, LSTM, and BERT.

| Model | Location | Steps |
|--------------|--------------------------|------------------------------------------|
| Naive Bayes | Notebooks/Models/NaiveBayes.ipynb | Run the notebook on Google Colab after uploading the required data files (combined_*.csv) to the root directory. This notebook takes about a minute or two to run. |
| LSTM | Notebooks/Models/LSTM.ipynb | Run the notebook on Google Colab after uploading the required data files (combined_*.csv) to the root directory. This notebook takes about 10 minutes to run. |
| BERT | Notebooks/Models/BERT.ipynb | Run the notebook on Google Colab after uploading the required data files (combined_*.csv) to the root directory. There are four cells that are commented out and should be uncommented depending on which data augmentation experiment you would like to run (None, Grover, Synonym Replacement, or Back Translation). Each experiment has its own additional required data files (indicated in the notebook) that need to be uploaded to the root directory. This notebook takes anywhere from 30 minutes to several hours depending on the experiment. |

## 5) Experiment Outputs

The experiments folder contains some of the PDF notebook outputs for the experiments we ran for model analysis and comparing different augmentation techniques.

### Note: Due to errors in saving from Google Colab, some of the outputs are cut off prematurely.

## 6) Web App

VerifAI is the tool we built to detect human fake news. It can be accessed at: https://shukieshah.github.io/VerifAI.

The code for the tool can be found at: https://github.com/shukieshah/VerifAI

We hope that VerifAI serves as a useful tool for the wider community. Please note that the tool is, by no means, perfect. The tool is not meant for neural fake news detection and sometimes classifies fake news as real. This is due to the wide variance of fake news in the real world that our model was not trained to detect. After all, this is precisely what makes reliable fake news detection such a difficult problem to solve!