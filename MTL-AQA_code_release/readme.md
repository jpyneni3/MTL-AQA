Author: Paritosh Parmar (https://github.com/ParitoshParmar)
Code used in the following, also if you find it useful, please consider citing the following:

@inproceedings{parmar2019and,
  title={What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment},
  author={Parmar, Paritosh and Tran Morris, Brendan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={304--313},
  year={2019}
}

Set the options in opts.py file, and train_test files.

Sports-1M pretrained C3D weights available from here: http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle



|  Experiment Number |  Description | Train  |  Test |  Model Folder |
|---|---|---|---| ---|
| 1 | C3DAVG model   |  c3davg_train_logging_file_1 | c3davg_test_logging_file_1  | c3davg_140_saved |
| 2 | C3DAVG model with SGD Backbone  |  train_logging_file_1 | test_logging_file_1  |c3davg_140_saved_s3d |  
| 3 | C3DAVG model with Attention  | c3d_attn_train_logging_file_1.txt   | c3d_attn_test_logging_file_1.txt  | c3davg_140_saved_attn  |
|4  |   |   |   | |