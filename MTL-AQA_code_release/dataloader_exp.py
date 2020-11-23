from torch.utils.data import DataLoader
from dataloaders.dataloader_C3DAVG import VideoDataset
import torch

train_batch_size = 3
test_batch_size = 5

train_dataset = VideoDataset('train')
test_dataset = VideoDataset('test')
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

for data in train_dataloader:
    # true_final_score = data['label_final_score'].unsqueeze_(1).type(torch.FloatTensor)
    # true_postion = data['label_position']
    # true_armstand = data['label_armstand']
    # true_rot_type = data['label_rot_type']
    # true_ss_no = data['label_ss_no']
    # true_tw_no = data['label_tw_no']
    true_captions = data['label_captions']
    print(true_captions)
    # true_captions_mask = data['label_captions_mask']
    # video = data['video'].transpose_(1, 2)

# for data in test_dataloader:
#     true_scores.extend(data['label_final_score'].data.numpy())
#     true_position.extend(data['label_position'].numpy())
#     true_armstand.extend(data['label_armstand'].numpy())
#     true_rot_type.extend(data['label_rot_type'].numpy())
#     true_ss_no.extend(data['label_ss_no'].numpy())
#     true_tw_no.extend(data['label_tw_no'].numpy())
#     video = data['video'].transpose_(1, 2)