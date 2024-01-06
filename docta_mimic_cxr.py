import argparse

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import torch

from docta.utils.config import Config
from docta.datasets import CXRDataset
from docta.core.preprocess import Preprocess
from docta.datasets.data_utils import load_embedding
from docta.apis import DetectLabel
from docta.core.report import Report
from docta.models import load

'''
CUDA_VISIBLE_DEVICES=1 python docta_mimic_cxr.py --nll
'''

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
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
config_pth = "./config/mimiccxr_{}_{}.py".format(args.model, args.method)
print('Using data {} and config: '.format(args.data)+config_pth)

cfg = Config.fromfile(config_pth)
cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.nll = args.nll

dataset_raw = CXRDataset(cfg)

train_indices, val_indices = train_test_split(np.arange(len(dataset_raw)),test_size=0.2)
train_pretrain_indices, train_baseline_indices = train_test_split(train_indices,test_size=0.5)
val_indices, test_indices = train_test_split(val_indices,test_size=0.5)
data_indices = np.append(train_baseline_indices, test_indices)

print("Step 2: Extract Embedding")
"""
Note: 
1. Strongly recommend to use a GPU to encode features.
2. The embedding will be automatically saved by running pre_processor.encode_feature()
"""
pre_processor = Preprocess(cfg, dataset_raw, data_indices=data_indices)

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

y_pred = np.array([i in idxs for i in range(len(dataset)-len(test_indices), len(dataset))])
y_true = dataset_raw.label[test_indices, 0] != dataset_raw.label[test_indices, 1]

print('predicted noisy label: {}, actual noisy label: {}'.format(y_pred.sum(), y_true.sum()))
print('F1: {}'.format(f1_score(y_true, y_pred)))