Set the options in opts.py file, and train_test files.

##Model Setup and References:

C3D Model[1]:  
Download the pretrained C3D weights [from here](http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle)  
Rename as c3d.pickle  
Provided by Parmar and Morris  

S3D Model[2]:
Download the pretrained S3D weights [from here] (https://drive.google.com/uc?export=download&id=1HJVDBOQpnTMDVUM3SsXLy0HUkf_wryGO)  
Rename as S3D_kinetics400.pt  
Provided by Kyle Min and Jason Corso [2]

LSTM Autoencoder[3]:
/models/C3DAVG/LSTM_autoencoder.py



[1]
```
@inproceedings{mtlaqa,
  title={What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment},
  author={Parmar, Paritosh and Tran Morris, Brendan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={304--313},
  year={2019}
}
```

[2]
```
@inproceedings{min2019tased,
  title={TASED-Net: Temporally-Aggregating Spatial Encoder-Decoder Network for Video Saliency Detection},
  author={Min, Kyle and Corso, Jason J},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={2394--2403},
  year={2019}
}
```
[3]
```
@article{DBLP:journals/corr/SrivastavaMS15,
  author    = {Nitish Srivastava and
               Elman Mansimov and
               Ruslan Salakhutdinov},
  title     = {Unsupervised Learning of Video Representations using LSTMs},
  journal   = {CoRR},
  volume    = {abs/1502.04681},
  year      = {2015},
  url       = {http://arxiv.org/abs/1502.04681},
  archivePrefix = {arXiv}
}
```

