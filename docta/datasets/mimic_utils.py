import numpy as np
import torch
import torchvision
import torch
from torchvision import transforms
import random

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm

from lib.datasets.image_dataloaders import ImageDataset
from lib.datasets.dataloader import NoisyCombinedDataset, CXRDataset, NoisyCombinedMultiModalDataset

cifar10_labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
cifar10_labels = np.array(cifar10_labels)

cifar100_labels = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]

cifar100_labels = np.array(cifar100_labels)

mimiccxr_labels = [
    "No finding",
    "Clinical finding"
]
mimiccxr_labels = np.array(mimiccxr_labels)

#NB: Resizing to 224x224 because several input networks (e.g. ViT) expects
# this size
generic_transform = transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def add_noisy_labels(dataset, name, generic_transform):
    
    if name=='cifar10':
        noise_label = torch.load('./data/CIFAR-10_human.pt')['worse_label']
    elif name=='cifar100':
        noise_label = torch.load('./data/CIFAR-100_human.pt')['noisy_label']
    elif name=='mimiccxr':
        raise NotImplementedError
    return noise_label

def get_dataset(name, data_seed, noisy_labels=False, percent_flips=0.20,
                flip_type=None, multimodal=False,
                return_combined_dataset=False):

    if name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=generic_transform
        )
        trainset_updated=trainset
        text_labels = cifar10_labels[trainset.targets]
        
        if noisy_labels:
            noise_labels = add_noisy_labels(trainset, name, transform)


    elif name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=generic_transform
        )
        trainset_updated=trainset
        text_labels = cifar100_labels[trainset.targets]
        if noisy_labels:
            noise_labels = add_noisy_labels(trainset, name, transform)

        
    elif name == 'mimiccxr':
        trainset = CXRDataset(data_path='/data/healthy-ml/gobi1/data/', 
                              split='tr', hparams=None, downsample=True)
        text_labels = mimiccxr_labels[np.array(trainset.y, dtype=np.int8)]
        if noisy_labels:
            raise NotImplementedError
        
    elif name == 'multimodal_mimiccxr':
        trainset = CXRDataset(data_path='/data/healthy-ml/gobi1/data/', 
                              split='tr', hparams=None, multimodal=True, downsample=True)
        text_labels = mimiccxr_labels[np.array(trainset.y, dtype=np.int8)]
        if noisy_labels:
            raise NotImplementedError

    else:
        raise NotImplementedError
    
    if not noisy_labels and return_combined_dataset:
        try:
            assert flip_type is not None
        except:
            raise NotImplementedError
        noise_labels, _ = get_label_flips(name, percent_flips, text_labels, flip_type)
        
    # NB: Data split is : 40% pretrain, 40% baseline/method training, 10% validation, 10% testing
    train_indices, val_indices = train_test_split(np.arange(len(trainset)),test_size=0.2, random_state=data_seed)
    train_pretrain_indices, train_baseline_indices = train_test_split(train_indices,test_size=0.5, random_state=data_seed)
    val_indices, test_indices = train_test_split(val_indices,test_size=0.5, random_state=data_seed)
    
    train_set_pretrain = torch.utils.data.Subset(trainset, train_pretrain_indices)
    train_set_baseline = torch.utils.data.Subset(trainset, train_baseline_indices)
    val_set = torch.utils.data.Subset(trainset, val_indices)
    test_set = torch.utils.data.Subset(trainset, test_indices)

    if noisy_labels and not return_combined_dataset:
        return train_set_pretrain, train_set_baseline, val_set, test_set, noise_labels[train_pretrain_indices], noise_labels[train_baseline_indices], noise_labels[val_indices], noise_labels[test_indices]
    elif return_combined_dataset:
        if multimodal:
                return NoisyCombinedMultiModalDataset(train_set_pretrain,noise_labels[train_pretrain_indices]), NoisyCombinedMultiModalDataset(train_set_baseline, noise_labels[train_baseline_indices]), NoisyCombinedMultiModalDataset(val_set, noise_labels[val_indices]), NoisyCombinedMultiModalDataset(test_set, noise_labels[test_indices])

        return NoisyCombinedDataset(train_set_pretrain,noise_labels[train_pretrain_indices]), NoisyCombinedDataset(train_set_baseline, noise_labels[train_baseline_indices]), NoisyCombinedDataset(val_set, noise_labels[val_indices]), NoisyCombinedDataset(test_set, noise_labels[test_indices])
    else:
        return train_set_pretrain, train_set_baseline, val_set, test_set

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
        if dataset == 'cifar10':
            label_set = list(cifar10_labels)
        elif dataset == 'cifar100':
            label_set = list(cifar100_labels)
        elif dataset == 'mimiccxr':
            label_set = list(mimiccxr_labels)
        else:
            raise NotImplementedError

        for label in all_label_values:
            curr_label_index = label_set.index(label)
            new_index = curr_label_index + 1
            if (curr_label_index + 1) >= len(label_set):
                new_index-=len(label_set)
            random_flip_dict[label] = label_set[new_index]
        flip_dict = random_flip_dict
        
        

    else:
        if dataset == "cifar100":
            realistic_label_flip_dict = {
                "baby": "girl",
                "bear": "tractor",
                "bowl": "castle",
                "boy": "man",
                "bridge": "plate",
                "bus": "streetcar",
                "can": "mushroom",
                "couch": "bed",
                "crab": "aquarium_fish",
                "fox": "bear",
                "girl": "woman",
                "leopard": "caterpillar",
                "lion": "caterpillar",
                "man": "woman",
                "mushroom": "pickup_truck",
                "oak_tree": "maple_tree",
                "orchid": "sunflower",
                "palm_tree": "maple_tree",
                "poppy": "sunflower",
                "shark": "aquarium_fish",
                "tiger": "caterpillar",
                "tractor": "pickup_truck",
                "trout": "aquarium_fish",
                "willow_tree": "maple_tree",
            }
        elif dataset == "cifar10":
            realistic_label_flip_dict = {
                "airplane": "bird",
                "bird": "airplane",
                "dog": "deer",
                "deer": "dog",
                "cat": "dog",
                "ship": "truck",
                "truck": "automobile"
            }
        else:
            raise NotImplementedError
        flip_dict = realistic_label_flip_dict

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
    return np.array(flipped_labels), np.array(flip_labels)