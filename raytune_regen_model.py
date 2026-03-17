# libraries
import os
import numpy as np
from raytune_utils import trainGCN, PatientDataset # custom dataset and trainer 
import pytorch_lightning as pl
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold
import argparse
import pandas as pd
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer
from ray.train import CheckpointConfig, ScalingConfig
# from dotenv import load_dotenv

# assert load_dotenv()

# reproducibility
pl.seed_everything(42)

# hyperparams
tot_epochs = 50 # 50
num_inp_ft = 1

# command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument("--adj_init", type=str, default='None', help="Adjacency matrix initialization: None, Spearman, Pearson, PPI, CPDB")
parser.add_argument("--label", type=str, default='vital_status', help="which label to classify") 
parser.add_argument("--cancer_type", type=str, default='coadread', help="what cancer type to use: coadread, brca, luad") 
parser.add_argument("--runID", type=str, required=True, help="Unique runID") 
args = parser.parse_args()

adj_init = args.adj_init
label_arg = args.label
cancer_type = args.cancer_type
runID = args.runID

def train_TCGA(config):
    # parameters from config
    bs = config['bs']
    lr = config['lr']
    conv_alg = config['conv_alg']
    cheb_filters = config['cheb_filters'] if conv_alg == 'cheb' else None
    num_heads = config['num_heads'] if conv_alg == 'gat' else None
    emb_size = config['emb_size']
    num_layers = config['num_layers']
    num_nodes = config['num_nodes']
    k = config['k']
    distance_metric = config['distance_metric']
    pooling_alg = config['pooling_alg']
    dropout_val = config['dropout_val']

    # generate the dataset
    dataset = PatientDataset(adj_init, label_arg, cancer_type)
    ew_mat = dataset.out_mat

    # do kfold cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    f1_foldwise = []
    acc_foldwise = []

    # split train, test with straitfy
    fold_num = 1
    for train_idx, test_idx in kfold.split(np.zeros(len(dataset)), dataset.labels):
        # split train_idx into train and validation in 90-10 ratio
        train_idx = list(train_idx)
        np.random.shuffle(train_idx)
        val_idx = train_idx[:int(0.1*len(train_idx))]
        train_idx = train_idx[int(0.1*len(train_idx)):]

        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        test_ds = Subset(dataset, test_idx)

        print("Fold Number: ", str(fold_num))
        # print("samples in train dataset: ", len(train_ds))
        # print("samples in validation dataset: ", len(val_ds))
        # print("samples in test dataset: ", len(test_ds))

        model_name = 'REGEN_' + str(cancer_type) + '_' + str(label_arg) + '_' + str(adj_init) + '_' + str(conv_alg) + '_' + str(cheb_filters) + '_' + str(num_heads) + '_' + str(emb_size) + '_' + str(num_layers) + '_' + str(num_nodes)+ '_' + str(bs)+ '_' + str(lr) + '_' + str(k) + '_' + str(distance_metric) + '_' + str(pooling_alg) + '_' + str(fold_num)
        # model parameters
        save_loc = '/path/saved_models/' + model_name

        # train model
        model, result = trainGCN(emb_size, num_layers, num_nodes, tot_epochs, bs, lr, save_loc, train_ds, val_ds, test_ds, dropout_val, conv_alg, cheb_filters, num_heads, fold_num, label_arg, dataset.pos_weight, dataset.num_genes, cancer_type, k, distance_metric, pooling_alg, ew_mat)

        f1_foldwise.append(result['test'][0]['test_f1'])
        acc_foldwise.append(result['test'][0]['test_acc'])

        fold_num += 1

    # print("Num Fold with Test F1 Scores: " + str(len(f1_foldwise)) + " out of 5 \n")
    # print("Final F1 scores (Mean +- STD): " + str(np.mean(f1_foldwise)) + " +- " + str(np.std(f1_foldwise)) + "\n")
    # print("Final Acc scores (Mean +- STD): " + str(np.mean(acc_foldwise)) + " +- " + str(np.std(acc_foldwise)) + "\n")

    # return the mean F1 score for hyperparameter tuning
    tune.report({'mean_f1': np.mean(f1_foldwise), 'std_f1': np.std(f1_foldwise), 'mean_acc': np.mean(acc_foldwise), 'std_acc': np.std(acc_foldwise)})


config = {
    "bs": tune.choice([1, 2, 4, 8, 16, 32]),
    "lr": tune.loguniform(1e-5, 1e-2),
    "conv_alg": tune.choice(['gcn', 'cheb', 'gat']), 
    # "conv_alg": tune.choice(['gcn', 'sage', 'cheb', 'gat', 'gin']),
    "cheb_filters": tune.choice([2, 4, 6, 8, 10]),
    "num_heads": tune.choice([4, 8]),
    "emb_size": tune.choice([16, 32, 64, 128]),
    "num_layers": tune.choice([2, 4, 6]),
    "num_nodes": tune.choice([32, 64, 128]),
    "k": tune.choice([3, 5, 7, 9, 11, 13]),
    "distance_metric": tune.choice(['euclidean', 'cosine']),
    "pooling_alg": tune.choice(['mean', 'max', 'add', 'topk', 'asap', 'flatten']),
    "dropout_val": tune.uniform(0.0, 0.3),
}
    

tuner = tune.Tuner(
    tune.with_resources(
        tune.with_parameters(train_TCGA),
        resources={"cpu": 0, "gpu": 0.1}
    ),
    tune_config=tune.TuneConfig(
        num_samples=100, # 100
        scheduler=ASHAScheduler(metric="mean_f1", 
                                mode="max", 
                                max_t=tot_epochs, 
                                grace_period=10, # 5
                                reduction_factor=2),
    ),
    run_config = tune.RunConfig(
        name = 'RT_' + cancer_type + '_' + adj_init + '_' + runID,
        storage_path= "/path/results/" #os.environ['TUNE_STORAGE_PATH']
    ),
    param_space=config,
)

results = tuner.fit()

# save in df
df = pd.DataFrame([result.metrics for result in results])
df.to_csv(f'/path/results/raytune_results_{adj_init}_{label_arg}_{cancer_type}.csv', index=False)

