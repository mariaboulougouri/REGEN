# libraries
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Dataset, Data
from torch_geometric.nn.pool import global_mean_pool, global_max_pool, global_add_pool, TopKPooling, ASAPooling
from torch_geometric.transforms import KNNGraph
from sklearn.metrics import classification_report, balanced_accuracy_score, roc_curve
import lightning as L
import math
import torchmetrics
from scipy import sparse
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import SAGEConv, ChebConv, GINConv
from torch.nn import ModuleList
import os

class PatientDataset(Dataset):
    def __init__(self, adj_init, label_arg, cancer_type): 
        super().__init__() 

        print("loading dataset")

        if cancer_type == 'coadread':
            folder = 'TCGA_COADREAD'
        elif cancer_type == 'brca':
            folder = 'TCGA_BRCA'
        elif cancer_type == 'hnsc':
            folder = 'TCGA_HNSC'
        elif cancer_type == 'kipan':
            folder = 'TCGA_KIPAN'
        elif cancer_type == 'luad':
            folder = 'TCGA_LUAD'
        elif cancer_type == 'stes':
            folder = 'TCGA_STES'
        elif cancer_type == 'gbmlgg':
            folder = 'TCGA_GBMLGG'

        if label_arg == 'vital_status':
            metadata = 'patient.vital_status' # one example
        elif label_arg == 'follow_up':
            metadata = 'patient.follow_ups.follow_up.vital_status' # one example
        elif label_arg == 'neoplasm':
            metadata = 'patient.person_neoplasm_cancer_status' # one example

        data_path = '/path/' + folder + '/'
        df_labels = pd.read_csv(data_path + "filtered_metaclinical.txt", sep= '\t', index_col=0)   
        df = pd.read_csv(data_path + "filtered_rnaseq_zerosremoved_lassoselectedc1000_logtransformed.txt", sep= '\t', index_col=0, header=0)

        df_labels = df_labels[metadata]
        df_labels = df_labels.dropna()
        df.index = df.index.str[:12].str.lower()
        df = df.loc[df_labels.index]
        df_labels = df_labels.astype('category')
        df_labels = df_labels.cat.codes
        self.labels = np.asarray(df_labels.values)
        print("Num Unique Labels: ", len(np.unique(self.labels)))
        self.num_outs = len(list(set(list(self.labels))))
        print("Num Unique Labels: ", self.num_outs)
        num_0 = np.sum(self.labels == 0) / len(self.labels)
        num_1 = np.sum(self.labels == 1) / len(self.labels)
        self.weights = [1/ num_0, 1/ num_1]
        self.pos_weight = [1/ num_0]

        self.features = np.asarray(df.values)

        self.num_genes = len(self.features[0])

        if adj_init == 'Pearson':
            # load adj matrix
            pearson_adj_matrix = df.corr(method='pearson')
            new_adj_matrix = pearson_adj_matrix.dropna(axis=0, how='all')
            new_adj_matrix = new_adj_matrix.dropna(axis=1, how='all')
            self.out_mat = new_adj_matrix.to_numpy()
            # nan to zero
            self.out_mat = np.abs(np.nan_to_num(self.out_mat, nan=0.0))
        elif adj_init == 'Spearman':
            spearman_adj_matrix = df.corr(method='spearman')
            new_adj_matrix = spearman_adj_matrix.dropna(axis=0, how='all')
            new_adj_matrix = new_adj_matrix.dropna(axis=1, how='all')
            self.out_mat = new_adj_matrix.to_numpy()
            self.out_mat = np.abs(np.nan_to_num(self.out_mat, nan=0.0))
        elif adj_init == 'PPI':
            ppi = np.load(data_path + f'/ppi_combined_patient.vital_status_logtrans.npz')
            self.out_mat = ppi['arr_0']
            self.out_mat = np.abs(np.nan_to_num(self.out_mat, nan=0.0))
        elif adj_init == 'None':
            self.out_mat = np.ones((self.num_genes, self.num_genes))
        elif adj_init == 'CPDB':
            # cpdb = np.load(data_path + f'/CPDB_pathways_{metadata}_logtrans.npz')
            cpdb = np.load(data_path + f'/CPDB_pathways_frequency_human_patient.vital_status_logtrans.npz')
            self.out_mat = cpdb['arr_0']
            self.out_mat = np.abs(np.nan_to_num(self.out_mat, nan=0.0))
        
        print("Num Genes: ", self.num_genes)

    def len(self):
        return len(self.features)
    
    def get(self, idx):
        y_out = torch.zeros(self.num_outs)
        y_out[self.labels[idx]] = 1

        x_ft = self.features[idx]
        data = Data(x = torch.tensor(x_ft, dtype = torch.float).unsqueeze(1), y = y_out)

        data.x = data.x.unsqueeze(0)

        data.num_outs = self.num_outs
        data.weights = self.weights
        data.pos_weight = self.pos_weight

        return data

class KNNGraphLearn(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        
        self.model_params=hparams
        num_layers = hparams['num_layers']
        self.num_nodes = hparams['num_nodes']
        self.emb_size = hparams['emb_size']
        self.conv_alg = hparams['conv_alg']
        self.cheb_filters = hparams['cheb_filters'] if self.conv_alg == 'cheb' else None
        self.num_heads = hparams['num_heads'] if self.conv_alg == 'gat' else None
        self.fold_num = hparams['fold_num']
        self.label_arg = hparams['label_arg']
        self.pos_weight = hparams['pos_weight']
        self.num_genes = hparams['num_genes'] 
        self.cancer_type = hparams['cancer_type']
        self.distance_metric = hparams['distance_metric']
        self.pooling_alg = hparams['pooling_alg']
        self.ew_mat = hparams['ew_mat']

        self.device_attr = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.dropout_layer = nn.Dropout(p=hparams['dropout'])

        self.input_mlp = torch.nn.Linear(1, self.emb_size) 

        self.gnn_model = ModuleList() 
        if self.conv_alg == 'gcn':
            self.gnn_model.append(GCNConv(self.emb_size, self.num_nodes))
        elif self.conv_alg == 'sage':
            self.gnn_model.append(SAGEConv(self.emb_size, self.num_nodes))
        elif self.conv_alg == 'cheb':
            self.gnn_model.append(ChebConv(self.emb_size, self.num_nodes, K=self.cheb_filters))
        elif self.conv_alg == 'gat':
            self.gnn_model.append(GATConv(self.emb_size, self.num_nodes, heads=self.num_heads))
        elif self.conv_alg == 'gin':
            self.gnn_model.append(GINConv(nn=nn.Linear(self.emb_size, self.num_nodes)))

        for _ in range(1, num_layers):
            if self.conv_alg == 'gcn':
                self.gnn_model.append(GCNConv(self.num_nodes, self.num_nodes))
            elif self.conv_alg == 'sage':
                self.gnn_model.append(SAGEConv(self.num_nodes, self.num_nodes))
            elif self.conv_alg == 'cheb':
                self.gnn_model.append(ChebConv(self.num_nodes, self.num_nodes, K=self.cheb_filters))
            elif self.conv_alg == 'gat':
                self.gnn_model.append(GATConv(self.num_nodes, self.num_nodes, heads=self.num_heads))
            elif self.conv_alg == 'gin':
                self.gnn_model.append(GINConv(nn=nn.Linear(self.num_nodes, self.num_nodes)))

        if self.pooling_alg == 'topk':
            self.pool = TopKPooling(self.num_nodes, ratio=0.5).to(self.device_attr)
        elif self.pooling_alg == 'asap':
            self.pool = ASAPooling(self.num_nodes, ratio=0.5).to(self.device_attr)
        
        if self.pooling_alg in ['mean', 'max', 'add']:
            self.output_mlp = torch.nn.Linear(self.num_nodes, 1)
        elif self.pooling_alg in ['topk', 'asap']:
            self.output_mlp = torch.nn.Linear(int(math.ceil(self.num_genes * 0.5)) * self.num_nodes, 1)
        elif self.pooling_alg == 'flatten':
            self.output_mlp = torch.nn.Linear(self.num_nodes * self.num_genes, 1)

        if self.distance_metric == 'euclidean':
            self.knn_graph_transform = KNNGraph(k=hparams['k'], force_undirected=True, loop=False, cosine=False)
        elif self.distance_metric == 'cosine':
            self.knn_graph_transform = KNNGraph(k=hparams['k'], force_undirected=True, loop=False, cosine=True)

        self.sigmoid = nn.Sigmoid()
        self.relu_act = nn.ReLU()
        self.accuracy = torchmetrics.Accuracy(task="binary")

        self.preds_list = []
        self.true_list = []
        self.val_loss = []
        
    def forward(self, x):
        # input mlp
        x = self.input_mlp(x)
        x = self.relu_act(x)
        x = self.dropout_layer(x)

        # knn graph
        d = Data(pos=x.squeeze(0)).to(self.device_attr)
        d = self.knn_graph_transform(d)

        ei = d.edge_index

        src, dst = ei

        ew = torch.tensor(self.ew_mat[src.cpu().numpy(), dst.cpu().numpy()]).to(self.device_attr).to(torch.float).unsqueeze(1)

        if self.conv_alg == 'gat':
            x = x.squeeze(0)

        for l in self.gnn_model:
            if self.conv_alg in ['sage', 'gin']:
                x = l(x, ei)
            else:
                x = l(x, ei, ew)
            x = self.relu_act(x)

            if self.conv_alg == 'gat':
                x = x.view(self.num_genes, self.num_nodes, self.num_heads)
                x = torch.mean(x, dim=-1)

            x = self.dropout_layer(x)

        if self.conv_alg == 'gat':
            x = x.unsqueeze(0)

        # pool outputs
        if self.pooling_alg == 'mean':
            x = global_mean_pool(x, d.batch)
        elif self.pooling_alg == 'max':
            x = global_max_pool(x, d.batch)
        elif self.pooling_alg == 'add':
            x = global_add_pool(x, d.batch)
        elif self.pooling_alg == 'topk':
            x = x.squeeze(0)
            x = self.pool(x, ei)[0]
            x = torch.flatten(x)
        elif self.pooling_alg == 'asap':
            x = x.squeeze(0)
            x = self.pool(x, ei)[0]
            x = torch.flatten(x)
        elif self.pooling_alg == 'flatten':
            x = torch.flatten(x)

        # linear out
        out = self.output_mlp(x)       
        if self.pooling_alg in ['add', 'mean', 'max']:
            out = out.squeeze(0)    

        out = self.dropout_layer(out)

        return out
   
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.model_params['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.model_params['num_epochs'], eta_min=0)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        # set model to train
        self.train()
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()
        
        data = train_batch
        X = data.x
        y = data.y
        y = torch.tensor([torch.argmax(y)]).cuda().float()
        
        pred = self(X)

        w_p = torch.FloatTensor([self.pos_weight]).cuda()

        pred = pred.unsqueeze(0)
        y = y.unsqueeze(0)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, y, pos_weight=w_p) 
        loss.backward()
        
        optimizer.step()

        self.log('train_loss', loss.detach().cpu())
    
    def validation_step(self, train_batch, batch_idx):
        # set model to eval
        self.eval()
        data = train_batch
        X = data.x
        y = data.y
        y = torch.tensor([torch.argmax(y)]).cuda().float()
        
        pred = self(X)
        self.preds_list.append(self.sigmoid(pred).detach().cpu().numpy()[0])

        w_p = torch.FloatTensor([self.pos_weight]).cuda()

        pred = pred.unsqueeze(0)
        y = y.unsqueeze(0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred,y, pos_weight=w_p)
        
        self.true_list.append(int(y.detach().cpu().numpy()[0]))
        
        self.log('val_loss', loss)

    def on_validation_epoch_end(self):
        fpr, tpr, thresholds = roc_curve(self.true_list, self.preds_list)
        best_thresh = thresholds[np.argmax(tpr - fpr)]
        # print("Best Thresh Val: ", best_thresh)
        self.preds_list = [1 if i >= best_thresh else 0 for i in self.preds_list]

        # make confusion matrix 
        cr = classification_report(self.true_list, self.preds_list, output_dict=True)
        f1_val = cr['weighted avg']['f1-score']
        acc = balanced_accuracy_score(self.true_list, self.preds_list)

        self.log('val_f1', f1_val)
        self.log('val_acc', acc)

        self.true_list = []
        self.preds_list = []

    def test_step(self, train_batch, batch_idx):
        # set model to eval
        self.eval()
        data = train_batch
        X = data.x
        y = data.y
        y = torch.tensor([torch.argmax(y)]).cuda().float()
     
        pred = self(X)
        self.preds_list.append(self.sigmoid(pred).detach().cpu().numpy()[0])

        w_p = torch.FloatTensor([self.pos_weight]).cuda()
        pred = pred.unsqueeze(0)
        y = y.unsqueeze(0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred,y, pos_weight=w_p)
        self.true_list.append(int(y.detach().cpu().numpy()[0]))

        self.log('test_loss', loss.detach().cpu())

    def on_test_epoch_end(self):
        fpr, tpr, thresholds = roc_curve(self.true_list, self.preds_list)
        best_thresh = thresholds[np.argmax(tpr - fpr)]
        # print("Best Thresh Test: ", best_thresh)
        self.preds_list = [1 if i >= best_thresh else 0 for i in self.preds_list]

        # make confusion matrix 
        cr = classification_report(self.true_list, self.preds_list, output_dict=True)
        f1_test = cr['weighted avg']['f1-score']
        acc = balanced_accuracy_score(self.true_list, self.preds_list)
        
        self.log('test_f1', f1_test)
        self.log('test_acc', acc)

        self.best_val_f1 = 0
        self.best_edges = None
        self.true_list = []
        self.preds_list = []

def distanceMatrix(model, train_loader, val_loader, test_loader):
    # Get the distance matrix from the model
    model.eval()
    all_data = []
    
    for loader in [train_loader, val_loader, test_loader]:
        for batch in loader:
            x = batch.x
            all_data.append(x)
    
    # Concatenate all data
    all_x = torch.cat(all_data, dim=0)

    # Get embeddings
    with torch.no_grad():
        all_x = all_x.to(model.device_attr)
        embeddings = model.input_mlp(all_x)

    embeddings = torch.mean(embeddings, dim=0)  # Average over the first dimension

    # Compute distance matrix
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # Euclidean distance

    return dist_matrix.cpu().numpy()

def trainGCN(emb_size, num_layers, num_nodes, num_epochs, bs, lr, save_loc, train_loader, val_loader, test_loader, dropout_val, conv_alg, cheb_filters, num_heads, fold_num, label_arg, pos_weight, num_genes, cancer_type, k, distance_metric, pooling_alg, ew_mat):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=save_loc,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=bs,
        max_epochs=num_epochs,
        num_sanity_val_steps=0,
        enable_progress_bar = False, # NEW
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(dirpath=save_loc,
                monitor='val_loss',
                save_top_k=1,
                mode='min'),
            L.pytorch.callbacks.LearningRateMonitor("epoch"),
            L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=10),
        ],
    )
    # trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    # trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    hparams = {}    
    hparams['conv_alg'] = conv_alg
    hparams['cheb_filters'] = cheb_filters
    hparams['num_heads'] = num_heads
    hparams['emb_size'] = emb_size
    hparams['num_layers'] = num_layers
    hparams['num_nodes'] = num_nodes
    hparams['dropout'] = dropout_val
    hparams['lr'] = lr
    hparams['distance'] = 'euclidean'
    hparams['num_epochs'] = num_epochs
    hparams['fold_num'] = fold_num
    hparams['label_arg'] = label_arg
    hparams['pos_weight'] = pos_weight
    hparams['num_genes'] = num_genes
    hparams['cancer_type'] = cancer_type
    hparams['k'] = k
    hparams['distance_metric'] = distance_metric
    hparams['pooling_alg'] = pooling_alg
    hparams['ew_mat'] = ew_mat
    model = KNNGraphLearn(hparams)

    # training
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)

    # testing
    test_result = trainer.test(model, dataloaders = test_loader, verbose = False, ckpt_path = "best")
    result = {"test": test_result}

    # find the name of the .ckpt file in the model save dir
    ckpt_files = [f for f in os.listdir(save_loc) if f.endswith('.ckpt')]
    best_ckpt = ckpt_files[0]

    # load the best model
    model = KNNGraphLearn.load_from_checkpoint(os.path.join(save_loc, best_ckpt), hparams=hparams).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    # get the distance matrix from the model
    dist_mat = distanceMatrix(model, train_loader, val_loader, test_loader)

    # save the distance matrix
    np.savez(os.path.join(save_loc, 'dist_matrix.npz'), dist_mat=dist_mat)
    # print("saved distance matrix")

    return model, result
    