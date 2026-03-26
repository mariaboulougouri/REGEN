# libraries
import numpy as np
import torch
import torch.nn as nn
import lightning as L
from torch.nn import Linear, Sequential
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.conv import GCNConv, SAGEConv, GINConv, GATv2Conv, ChebConv
from torch.autograd import Variable
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score, roc_curve

# num_heads = 4
K_const = 8

class GCNOnly(L.LightningModule):                                            
    def __init__(self, gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, algo, pos_weight, num_genes,f):
        super().__init__()
        self.module_list = nn.ModuleList()
        self.gcn_layers = gcn_layers
        self.module_list.append(GCNConv(num_inp_ft, gcn_layers[0]))
        for i in range(len(gcn_layers)-1):
            self.module_list.append(GCNConv(gcn_layers[i], gcn_layers[i+1]))
        self.dropout = nn.Dropout(dropout_val)
        self.linear = nn.Linear(num_genes * gcn_layers[-1], 1)
        self.num_genes = num_genes
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.pos_weight = pos_weight
        self.sigmoid = nn.Sigmoid()
        self.lr = lr
        self.bs = bs
        self.num_epochs = num_epochs
        self.algo = algo
        self.f = f
        self.true_list = []
        self.preds_list = []
        self.val_loss = [] 

    def forward(self, batch):
        x = batch.x
        ei = batch.edge_index
        for i in range(len(self.module_list)):
            x = self.module_list[i](x, ei)
            x = self.relu(x)
            x = self.dropout(x)
        x = x.view(-1, self.num_genes * self.gcn_layers[-1])
        out = self.linear(x)
        out = out.squeeze(0)
        return out
    
    def _get_loss(self, batch):
        y = batch.y
        y = torch.argmax(y, dim=0)
        y = y.unsqueeze(0)
        y = y.float()
        # print("True label for loss: ", y)
        y_pred = self.forward(batch)
        # print("Pred label for loss: ", y_pred)
        w_p = torch.FloatTensor([self.pos_weight]).cuda()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y, pos_weight=w_p)
        return loss, y, y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        monitor = 'val_loss'
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
    
    def training_step(self, batch):
        loss, y, y_pred = self._get_loss(batch)
        # print("True label for loss (train step): ", y)
        # print("Pred label for loss (train step): ", y_pred)
        self.log('train_loss', loss, batch_size=self.bs)
        return loss
    
    def validation_step(self, batch):
        loss, y, y_pred = self._get_loss(batch)
        self.preds_list.append(y_pred.detach().cpu().numpy()[0])
        self.true_list.append(int(y.detach().cpu().numpy()[0]))
        self.log('val_loss', loss)
        self.val_loss.append(loss.item())
        return loss
    
    def on_validation_epoch_end(self):
        fpr, tpr, thresholds = roc_curve(self.true_list, self.preds_list)
        best_thresh = thresholds[np.argmax(tpr - fpr)]
        print("Best Thresh Val: ", best_thresh)
        self.preds_list = [1 if i >= best_thresh else 0 for i in self.preds_list]
        print("HERE: ", self.preds_list)
        print(".  HERE: ", self.true_list)
        cr = classification_report(self.true_list, self.preds_list, output_dict=True)
        f1_weighted_val = cr['weighted avg']['f1-score']
        f1_macro_val = cr['macro avg']['f1-score']
        acc = balanced_accuracy_score(self.true_list, self.preds_list)
        print(f"Validation WF1: {f1_weighted_val}, Validation MF1: {f1_macro_val},Validation Accuracy: {acc}")
        val_loss_mean = np.mean(self.val_loss) 
        self.val_loss = []  
        self.f.write(" Val WF1: " + str(f1_weighted_val) + " Val MF1: " + str(f1_macro_val) + " Val Acc: " + str(acc) + " Val Loss: " + str(val_loss_mean) + "\n") 
        self.true_list = []
        self.preds_list = []
    
    def test_step(self, batch):
        loss, y, y_pred = self._get_loss(batch)
        self.preds_list.append(y_pred.detach().cpu().numpy()[0])
        self.true_list.append(int(y.detach().cpu().numpy()[0]))
        self.log('test_loss', loss)
        return loss
    
    def on_test_epoch_end(self):
        fpr, tpr, thresholds = roc_curve(self.true_list, self.preds_list)
        best_thresh = thresholds[np.argmax(tpr - fpr)]
        print("Best Thresh Val: ", best_thresh)
        self.preds_list = [1 if i >= best_thresh else 0 for i in self.preds_list]
        # print("HERE: ", self.preds_list)
        # print(".  HERE: ", self.true_list)
        cr = classification_report(self.true_list, self.preds_list, output_dict=True)
        f1_weighted_test = cr['weighted avg']['f1-score']
        f1_macro_test = cr['macro avg']['f1-score']
        acc = balanced_accuracy_score(self.true_list, self.preds_list)
        print(f"Test WF1: {f1_weighted_test}, Test MF1: {f1_macro_test}, Test Accuracy: {acc}")
        self.log('test_wf1', f1_weighted_test)
        self.log('test_mf1', f1_macro_test)
        self.log('test_acc', acc)
        self.f.write(" Test WF1: " + str(f1_weighted_test) +" Test MF1: " + str(f1_macro_test) + " Test Acc: " + str(acc) + "\n")
        self.true_list = [] 
        self.preds_list = []

def trainGCN(gcn_layers, num_epochs, bs, lr, train_loader, val_loader, test_loader, dropout_val, num_inp_ft, algo, seed, save_loc, pos_weight, num_genes,f):
    trainer = L.Trainer(
        default_root_dir=save_loc,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=bs,
        max_epochs=num_epochs,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(dirpath=save_loc,
                monitor='val_loss',
                save_top_k=2),
            L.pytorch.callbacks.LearningRateMonitor("epoch"),
            L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=10),
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    model = GCNOnly(gcn_layers=gcn_layers, dropout_val=dropout_val, num_epochs=num_epochs, bs=bs, lr=lr, num_inp_ft=num_inp_ft, algo=algo, pos_weight=pos_weight, num_genes=num_genes, f = f)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)
    test_result = trainer.test(model, dataloaders = test_loader, verbose = False, ckpt_path = "best")
    result = {"test": test_result}

    return model, result