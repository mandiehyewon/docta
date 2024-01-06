# dataset settings
seed = 0
dataset_type = 'CIFAR'
modality = 'image' # image, text, tabular
num_classes = 100
data_root = './data/cifar/'
data_preproc = 'generic'
label_path = './data/cifar/CIFAR-100_human.pt'
noisy_label_key = 'noisy_label'
clean_label_key = 'clean_label'
label_sel = 1 # which label/attribute we want to diagnose
train_label_sel = label_sel # 1 for noisy
test_label_sel = train_label_sel

file_name = 'c100_VITB_Vote'
dataset_type += '_' + file_name
save_path = f'./results/{dataset_type}/VITB_Vote/'

feature_type = 'embedding' 

embedding_model = 'hf-hub:laion/openai/clip-vit-base-patch32'
embedding_cfg = dict(
    save_num = 50,
    shuffle = False,
    batch_size = 1024,
    num_workers = 1,
)


accuracy = dict(topk = 1, threth = 0.5)
n_epoch = 10
print_freq = 390
details = False

train_cfg = dict(
    shuffle = False,
    batch_size = 128,
    num_workers = 1,
)

test_cfg = dict(
    shuffle = False,
    batch_size = 128,
    num_workers = 1,
)


optimizer = dict(
    name = 'SGD',
    config = dict(
        lr = 0.1    
    )
)


hoc_cfg = dict(
    max_step = 1501, 
    T0 = None, 
    p0 = None, 
    lr = 0.1, 
    num_rounds = 50, 
    sample_size = 35000,
    already_2nn = False,
    device = 'cpu'
)


detect_cfg = dict(
    num_epoch = 21,
    sample_size = 35000,
    k = 10,
    name = 'simifeat',
    method = 'mv'
)