import os
import random
import torch
import numpy as np
import pandas as pd
import numpy as np
from PIL import Image,ImageFile
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms


mimiccxr_labels = [
    "No finding",
    "Clinical finding"
]

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_label_flips(dataset, percent_flips, text_labels, flip_type="random"):

    print(len(text_labels))
    if flip_type == "random":
        random_flip_dict = {}
        all_label_values = list(set(text_labels))
        for label in all_label_values:
            not_label = list(set(text_labels) - set([label]))
            x = random.sample(not_label, 1)
            random_flip_dict[label] = x[0]
        flip_dict = random_flip_dict
        
    elif flip_type == "pseudorandom":
        random_flip_dict = {}
        all_label_values = list(set(text_labels))
        label_set = list(mimiccxr_labels)

        for label in all_label_values:
            curr_label_index = label_set.index(label)
            new_index = curr_label_index + 1
            if (curr_label_index + 1) >= len(label_set):
                new_index-=len(label_set)
            random_flip_dict[label] = label_set[new_index]
        flip_dict = random_flip_dict
        

    flip_labels = []
    flipped_labels = []
    n_labels = len(text_labels)
    flip_labels = np.zeros((n_labels,1)).squeeze()
    flip_idx = random.sample(list(np.arange(n_labels)), int(percent_flips*n_labels))
    flip_labels[flip_idx]=1

    for i,curr_label in tqdm(enumerate(text_labels)):
        if flip_labels[i]==1 and curr_label in flip_dict:
            curr_label=flip_dict[curr_label]
        flipped_labels.append(curr_label)
    assert len(flipped_labels)==n_labels
    return np.array(flipped_labels)


class BaseDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    INPUT_SHAPE = None       # Subclasses should override
    SPLITS = {               # Default, subclasses may override
        'tr': 'train',
        'va': 'validate',
        'te': 'test'
    }
    EVAL_SPLITS = ['te']     # Default, subclasses may override

    def __init__(self, root, split, split_path, label_path, metadata, transform, multimodal=False,patient_file_path='/data/healthy-ml/scratch/aparnab/MultimodalDiscordance/data/patients.csv.gz'):
        df_patient = pd.read_csv(patient_file_path)
        df_patient = df_patient.drop_duplicates(subset='subject_id')
        
        df = pd.read_csv(metadata)
        df_split = pd.read_csv(split_path)
        df_label = pd.read_csv(label_path)[['subject_id','study_id','No Finding']]
        df_label.loc[df_label['No Finding'].isna(),'No Finding']=0
        
        # merging with split info
        df=df.merge(df_split, on=['subject_id','study_id','dicom_id'])
        
        # merging with labels
        df=df.merge(df_label, on=['subject_id','study_id'])
        df=df.merge(df_patient[['subject_id','gender']], on='subject_id')
        
        df = df[df["split"] == (self.SPLITS[split])]
        df["filename"] = df.apply(lambda row: 'p{}/'.format(str(row.subject_id)[:2])+'p{}/'.format(
            row.subject_id) + 's{}/'.format(row.study_id)+'{}.jpg'.format(row.dicom_id), axis=1)
       
        self.idx = list(range(len(df)))
        self.x = df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        self.y = df["No Finding"].tolist()
        self.transform_ = transform
        self.multimodal = multimodal

        if self.multimodal:
            df["reportfilename"] = df.apply(lambda row: 'p{}/'.format(str(row.subject_id)[:2])+'p{}/'.format(row.subject_id) + 's{}.txt'.format(row.study_id), axis=1)
        self.df = df


    def __getitem__(self, index):
        i = self.idx[index]
        x = self.transform(self.x[i])            
        y = torch.tensor(self.y[i], dtype=torch.long)
        self.targets = y
        
        # y = np.stack((self.targets, self.noisy_label)).transpose()

        if self.multimodal:
            with open(self.df.iloc[i]['reportfilename'],'r') as f:
                report = f.read()
            return x, y, report
        return x, y, index

    def __len__(self):
        return len(self.idx)
    
class CXRDataset(BaseDataset):
    N_STEPS = 20001
    CHECKPOINT_FREQ = 1000
    N_WORKERS = 16
    INPUT_SHAPE = (3, 224, 224,)
    SPLITS = {               # Default, subclasses may override
        'tr': 'train',
        'va': 'validate',
        'te': 'test'
    }
    EVAL_SPLITS = ['te']

    def __init__(self, data_path, split, hparams, downsample, multimodal=False):
        metadata = os.path.join(data_path, "MIMIC-CXR-JPG", 'mimic-cxr-2.0.0-metadata.csv.gz')
        label_file =os.path.join(data_path, "MIMIC-CXR-JPG", 'mimic-cxr-2.0.0-chexpert.csv.gz')
        split_path = os.path.join(data_path, "MIMIC-CXR-JPG", 'mimic-cxr-2.0.0-split.csv.gz')
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.data_transform = transform
        self.data_type = "images"
        self.downsample = downsample
        self.data_root = data_path
        super().__init__('/', split, split_path, label_file, metadata, transform, multimodal)

    def load_label(self):
        text_labels = mimiccxr_labels[np.array(self.targets, dtype=np.int8)]
        self.noisy_label = get_label_flips(name= 'mimiccxr', percent_flips=0.20, text_labels=text_labels, flip_type='pseudorandom')
        self.label = np.stack((self.targets, self.noisy_label)).transpose()

    def transform(self, x):
        if self.downsample:
            reduced_img_path = list(Path(x).parts)

            try:
                reduced_img_path[-5] = 'downsampled_files'
                reduced_img_path = Path(*reduced_img_path).with_suffix('.png')
                reduced_img_path = os.path.join(self.data_root, 'MIMIC-CXR-JPG', reduced_img_path)
                x = reduced_img_path
                img = Image.open(x).convert("RGB").resize((224, 224))
            except:
                reduced_img_path = list(Path(x).parts)
                print('Using non-downsampled image for this one!')
                reduced_img_path[-5] = 'files'
                reduced_img_path = Path(*reduced_img_path).with_suffix('.jpg')
                reduced_img_path = os.path.join(self.data_root, 'MIMIC-CXR-JPG', reduced_img_path)
                x = reduced_img_path
                img = Image.open(x).convert("RGB").resize((224, 224))
                
        else:
            img_path = list(Path(x).parts)
            img_path[-5] = 'files'
            img_path = Path(*img_path).with_suffix('.jpg')
            img_path = os.path.join(self.data_root, 'MIMIC-CXR-JPG', img_path)
            x = img_path
            img = Image.open(x).convert("RGB").resize((224, 224))

        return self.data_transform(img)