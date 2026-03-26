from gnn_utils import trainGCN # custom dataset and trainer
import pytorch_lightning as pl
import numpy as np
from utils import PatientDataset
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold
import argparse
print("Libraries loaded")

# batch_size = 4
# lr = 1e-4
algo = 'GCN'
dropout_val = 0.0
seed = 1
num_inp_ft = 1
print("Parameters loaded")

# reproducibility
pl.seed_everything(seed)
tot_epochs = 100
# adj_init = 'Spearman'
# cancer_type = 'kipan'
# label_arg = 'neoplasm'
parser = argparse.ArgumentParser()
parser.add_argument("--cancer_type", type=str, default="stes", help="cancer_type")
parser.add_argument("--label_arg", type=str, default="vital_status", help="metadata")
parser.add_argument("--adj_init", type=str, default="REGEN", help="Spearman, Pearson, Zeros, Ones")
parser.add_argument("--corrth", type=float, default=1, help="No. of stdevs to threshold adj. matrix")

parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--bs", type=int, default=2, help="Batch size")
parser.add_argument("--num_nodes", type=int, default=64, help="No. of nodes")

args = parser.parse_args()
adj_init = args.adj_init
cancer_type = args.cancer_type
label_arg = args.label_arg
corrth = args.corrth
lr = args.lr
bs = args.bs
num_nodes = args.num_nodes
print(num_nodes)
gcn_layers = [num_nodes, num_nodes]

dataset = PatientDataset(adj_init, label_arg, cancer_type, corrth)
sample = dataset[0]
num_genes = sample.x.shape[1]
initial_adj = sample.edge_index

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
wf1_foldwise = []
mf1_foldwise = []
acc_foldwise = []
fold_num = 1
out_file_name = '/path/output/'+ str(algo) + '_' + str(cancer_type) + '_' + str(label_arg) + '_' + str(adj_init) +'_'+ str(corrth) +'_'+ str(lr) +'_'+ str(bs) +'_'+ str(num_nodes) + '.txt' ## CHANGE HERE
f = open(out_file_name, 'w')
for train_idx, test_idx in kfold.split(np.zeros(len(dataset)), dataset.labels):
    train_idx = list(train_idx)
    np.random.seed(1)
    np.random.shuffle(train_idx)
    val_idx = train_idx[:int(0.1*len(train_idx))]
    train_idx = train_idx[int(0.1*len(train_idx)):]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    print("Fold Number: ", str(fold_num))
    print("samples in train dataset: ", len(train_ds))
    print("samples in validation dataset: ", len(val_ds))
    print("samples in test dataset: ", len(test_ds))

    full_labels = []
    for k in range(len(test_ds)):
        full_labels.append(test_ds[k].y)
    for k in range(len(val_ds)):
        full_labels.append(val_ds[k].y) 
    for k in range(len(train_ds)):
        full_labels.append(train_ds[k].y)

    print("percentage of samples in dataset that are positive: ", sum(full_labels)/len(full_labels))
    pos_weight = (1 / (sum(full_labels)/len(full_labels)))
    pos_weight = pos_weight[1] # NEW
    print("Positive weight: ", pos_weight)

    model_name = 'Model: ' + str(algo) + ' Cancer type: ' + str(args.cancer_type) + ' Label: ' + str(label_arg) + ' Init: ' + str(adj_init) + ' Params: [LR:' + str(lr) + ', BS: ' + str(bs) + ', CT: ' + str(corrth) +  ', NN: ' + str(64) + ', FN: ' + str(fold_num) + ']'

    save_loc = '/path/saved_models/' + model_name

    model, result = trainGCN(gcn_layers, tot_epochs, bs, lr, train_ds, val_ds, test_ds, dropout_val, num_inp_ft, algo, seed, save_loc, pos_weight, num_genes, f)

    wf1_foldwise.append(result['test'][0]['test_wf1'])
    mf1_foldwise.append(result['test'][0]['test_mf1'])
    acc_foldwise.append(result['test'][0]['test_acc'])

    fold_num += 1

f.write("Num Fold with Test WF1 Scores: " + str(len(wf1_foldwise)) + " out of 5")
f.write("Num Fold with Test MF1 Scores: " + str(len(mf1_foldwise)) + " out of 5")
f.write("Final WF1 scores (Mean +- STD): " + str(np.mean(wf1_foldwise)) + " +- " + str(np.std(wf1_foldwise)))
f.write("Final MF1 scores (Mean +- STD): " + str(np.mean(mf1_foldwise)) + " +- " + str(np.std(mf1_foldwise)))
f.write("Final Acc scores (Mean +- STD): " + str(np.mean(acc_foldwise)) + " +- " + str(np.std(acc_foldwise)))
f.close()