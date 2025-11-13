import csv
import os
import argparse

import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch_geometric.loader import DataLoader
from sklearn import metrics
from model import CrypKANet
from data import ProDataset

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "ckpt"


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for Evaluation CrypKANet")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--esm_out", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--pe_dim", type=int, default=32)
    parser.add_argument("--pe_ratio", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attn_dropout", type=float, default=0.7)
    parser.add_argument("--act", type=str, default="ReLU")
    parser.add_argument("--weight", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.34)
    parser.add_argument("--beta", type=float, default=0.46)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--use_esm", action="store_true", default=True)
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--esm_name", type=str, default='esmc_600M')

    args = parser.parse_args()
    return args

def evaluate(use_esm, model, data_loader):
    model.eval()
    epoch_loss = 0.0

    valid_pred = []
    valid_true = []

    for data in data_loader:
        with torch.no_grad():
            data = data.to(device)

            x = data.x
            xyz_feats = data.xyz_feats
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            pe = data.pe
            batch = data.batch
            y_true = data.y
            
            if use_esm:
                esm = data.esm_feat
                y_pred = model(x, xyz_feats, edge_index, edge_attr, batch, pe, esm)
            else:
                y_pred = model(x, xyz_feats, edge_index, edge_attr, batch, pe)

            # calculate loss
            loss = model.criterion(y_pred, y_true)
            
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)

            epoch_loss += loss.item()         

    epoch_loss_avg = epoch_loss / len(data_loader)
    return epoch_loss_avg, valid_true, valid_pred

def analysis(y_true, y_pred, best_threshold = None):
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    # Calculate TPR and FPR
    fpr, tpr, _ = metrics.roc_curve(binary_true, y_pred)
    tpr_fpr_data = list(zip(tpr, fpr))
    pre_rec_data = list(zip(precisions,recalls))

    # Save TPR and FPR to a CSV file
    with open('result/auc_data/auc.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['TPR', 'FPR'])  # Header row
        writer.writerows(tpr_fpr_data)

    # Save PRC and REC to a CSV file
    with open('result/auc_data/mean_prc.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['PRE', 'REC'])  # Header row
        writer.writerows(pre_rec_data)

    results = {
        'binary_pred': binary_pred,
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results


def test(args, test_loader):
    # 计算正负样本的权重
    all_labels = []
    for graph in test_loader:
        labels = graph.y.cpu().numpy()
        all_labels.extend(labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    model = CrypKANet(
        channels=args.hidden_dim,
        heads=args.num_heads,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        act=args.act,
        pe_dim=args.pe_dim,
        pe_ratio=args.pe_ratio,
        esm_out=args.esm_out,
        num_layers=args.num_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        weight=args.weight,
        class_weights=class_weights,
        alpha=args.alpha,
        beta=args.beta,
        use_esm=args.use_esm,
        esm_dim=args.esm_dim,
    ).to(device)

    model_name = 'best_model.pth'
    print(model_name)
    model.load_state_dict(torch.load(os.path.join(model_path, model_name), map_location='cuda:0'))
    epoch_loss_test_avg, test_true, test_pred = evaluate(True, model, test_loader)
    result_test = analysis(test_true, test_pred, best_threshold=0.85)

    print("========== Evaluate Test set ==========")
    print("Test loss: ", epoch_loss_test_avg)
    print("Test binary acc: ", result_test['binary_acc'])
    print("Test precision:", result_test['precision'])
    print("Test recall: ", result_test['recall'])
    print("Test f1: ", result_test['f1'])
    print("Test AUC: ", result_test['AUC'])
    print("Test AUPRC: ", result_test['AUPRC'])
    print("Test mcc: ", result_test['mcc'])
    print("Threshold: ", result_test['threshold'])

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = parse_args()
    print(args)    
    set_seed(args.seed)
    
    if args.esm_name == "esm2_t6_8M":
        args.esm_dim = 320
    elif args.esm_name == "esm2_t12_35M":
        args.esm_dim = 480
    elif args.esm_name == "esm2_t30_150M":
        args.esm_dim = 640
    elif args.esm_name == "esm2_t33_650M":
        args.esm_dim = 1280
    elif args.esm_name == "esm2_t36_3B":
        args.esm_dim = 2560
    elif args.esm_name == "esm2_t48_15B":
        args.esm_dim = 5120
    elif args.esm_name == "esm1b_t33_650M":
        args.esm_dim = 1280
    elif args.esm_name == "esm1v_t33_650M":
        args.esm_dim = 1280
    elif args.esm_name == "esmc_600M":
        args.esm_dim = 1152
    else:
        args.esm_dim = 1280


    print("Creating Test set FULL ...")
    test_set = ProDataset(
        pe_dim=args.pe_dim,
        dataset_name="test_ALL",
    )

    print("len of test:", len(test_set))

    print("Evaluating on Test FULL ...")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    test(args, test_loader)

