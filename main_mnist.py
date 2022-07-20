import torch
import random
import numpy as np
from pathlib import Path

import utils.datasets as dl
from utils.visual_counterfactual_generation import targeted_translations

# device
is_cuda = torch.cuda.is_available()
device = torch.device('cuda') if is_cuda else torch.device('cpu')

# fix random seed
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# variables
epochs = 5
batch_size = 64*2
dataset = 'mnist'

# train and test dataloaders
train_loader = dl.MNIST(train=True, batch_size=batch_size, augm_flag=True)

test_loader = dl.MNIST(train=False, batch_size=batch_size, augm_flag=True)


relative_path = Path(__file__).resolve().parents[2].absolute()
path_saved_model = 'online_git_projects/InNOutRobustness/MNIST_models/' + \
    'ResNet18/AdvACET_19-07-2022_13:32:23'
folder = relative_path / path_saved_model

model_descriptions = [
    ('resnet18', folder, 'best', 1, False)
]

img_size = 28
in_dataset = torch.unsqueeze(test_loader.dataset.data[:10], dim=1)
in_labels = test_loader.dataset.targets[:10]


imgs = torch.unsqueeze(test_loader.dataset.data[:10], dim=1)
# target_list = [[t] for t in in_labels[:10]]
target_list = [[i for i in range(10) if i != t] for t in in_labels[:10]]
filenames = None
device_ids = None
eval_dir = Path('')

norm = 'L1.5'
radii = [50, 75, 100]
print('num images', len(imgs), len(target_list), batch_size)
targeted_translations(model_descriptions, radii, imgs, target_list,
                      batch_size, in_labels, device, eval_dir, dataset,
                      norm='l1.5', steps=75, attack_type='afw',
                      filenames=filenames, device_ids=device_ids)
