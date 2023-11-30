import argparse

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import torch

from docta.utils.config import Config
from docta.datasets import Cifar10_noisy, CIFAR_Sampler
from docta.core.preprocess import Preprocess
from docta.datasets.data_utils import load_embedding
from docta.apis import DetectLabel
from docta.core.report import Report
from docta.models import load

'''
CUDA_VISIBLE_DEVICES=1 python docta_cifar10.py --nll
'''

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default='VITB', type=str, help='embedding model')
parser.add_argument('--method', default='Rank', type=str, help='SimiFeat method', choices=['Rank', 'Vote'])
parser.add_argument('--data', default=None, type=str, help='dataset, None uses the preprocessing in the Docta repo')
parser.add_argument('--nll', action="store_true", help='use nll loss') # default is cosine similarity
parser.add_argument('--duplicate', action="store_true", default=False, help='use duplicate')
args = parser.parse_args()

print("Step 1: Load Dataset")
"""
Note: 
1. Please set data_root in the config file appropriately.
2. Download the data to data_root beforing running
"""
config_pth = "./config/cifar10_{}_{}.py".format(args.model, args.method)
print('Using data {} and config: '.format(args.data)+config_pth)
# VITH_Rank: F1 0.7959854232630384, 0.79648871372268, 0.7937296197684506, 0.7958775205377148, 0.7987601335240821
# VITH_Vote: F1 0.9215372087525283, 0.9207159778889181, 0.9214293232291338, 0.9217418744083307, 0.9199052132701421
# VITB_Rank F1 0.7616164673830785, 0.7622826679590476, 0.7596296858400364, 0.762345959366579, 0.7621413639531008
# VITB_Rank with duplicate=False and noisy label from the authors:0.859402908905333, 0.8580739983385519, 0.8561560601215221, 0.859571748162352, 0.8597868402578339
# VITB_Vote F1 0.8858046472725372, 0.8844007858546169, 0.8845750812113592, 0.8853488189594286, 0.8841763535448738
# VITB_Vote with duplicate=False and noisy label from the authors: 0.8555575879519541, 0.8559358354252128, 0.8562881562881562, 0.8545943304007819, 0.8549954170485793

cfg = Config.fromfile(config_pth)
cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.nll = args.nll
if args.data == 'ours':
    cfg.data_preproc = args.data
elif args.data == 'simifeat':
    cfg.label_path = './data/cifar/noise_label_human.pt'
    cfg.noisy_label_key = 'noise_label_train'
    cfg.clean_label_key = 'clean_label_train'
dataset_raw = Cifar10_noisy(cfg)

train_indices, val_indices = train_test_split(np.arange(len(dataset_raw)),test_size=0.2)
train_pretrain_indices, train_baseline_indices = train_test_split(train_indices,test_size=0.5)
val_indices, test_indices = train_test_split(val_indices,test_size=0.5)

sampler = CIFAR_Sampler(test_indices)

print("Step 2: Extract Embedding")
"""
Note: 
1. Strongly recommend to use a GPU to encode features.
2. The embedding will be automatically saved by running pre_processor.encode_feature()
"""
pre_processor = Preprocess(cfg, dataset_raw)

if cfg.embedding_model == 'hf-hub:laion/openai/clip-vit-base-patch32':
    model, preprocess = load("ViT-B/32", jit=False)
    pre_processor.encode_feature(model_embedding=model, preprocess=preprocess)
else: # VIT-H
    pre_processor.encode_feature()

# load embedding
data_path = lambda x: cfg.save_path + f'embedded_{cfg.dataset_type}_{x}.pt'
# Duplicate should be false unless we have a representation of dataset that was not well extracted (e.g., Jigsaw or Anthropic Read-Team data in the SimiFeat paper)
dataset, _ = load_embedding(pre_processor.save_ckpt_idx, data_path, duplicate=args.duplicate) 

print("Step 3: Generate & Save Diagnose Report")

# Initialize report
report = Report() 
# Detect human annotation errors. You can do this with only CPU.
detector = DetectLabel(cfg, dataset, report=report)
detector.detect()
# Save report
report_path = cfg.save_path + f'{cfg.dataset_type}_report.pt'
torch.save(report, report_path)
print(report.diagnose) # Printing out noise transition matrix

label_error = np.array(report.detection['label_error'])
idxs  = label_error[:, 0].astype(int) # indexes of the noisy labels
noisy_prob  = label_error[:, 1] # noisy label probability (averaged over #epoch runs)
sel_noisy_summary = np.round(noisy_prob).astype(bool)
print(f'[SimiFeat] We find {np.sum(sel_noisy_summary)} corrupted instances from {sel_noisy_summary.shape[0]} instances')

y_pred = np.array([i in idxs for i in range(50_000)])
y_true = dataset_raw.label[:, 0] != dataset_raw.label[:, 1]

print('predicted noisy label: {}, actual noisy label: {}'.format(y_pred.sum(), y_true.sum()))
print('F1: {}'.format(f1_score(y_true, y_pred)))