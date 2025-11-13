import os
import time
import argparse

import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch_geometric.loader import DataLoader
from sklearn import metrics
from sklearn.model_selection import KFold
from tqdm import tqdm
from datetime import datetime

from model import CrypKANet
from data import ProDataset, get_train_validation_data_loaders

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_time():
    # 获取当前日期和时间
    current_datetime = datetime.now()

    # 格式化日期和时间
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H:%M:%S")

    return formatted_datetime

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
    parser.add_argument("--alpha", type=float, default=0.36)
    parser.add_argument("--beta", type=float, default=0.97)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--use_esm", action="store_true", default=True)
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--esm_name", type=str, default='esmc_600M')

    args = parser.parse_args()
    return args

def train_one_epoch(use_esm, model, data_loader):
    model.train()
    epoch_loss_train = 0.0

    pbar = tqdm(total=len(data_loader), desc='Training')
    for data in data_loader:
        pbar.update(1)
        model.optimizer.zero_grad()
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

        def closure():
            loss = model.criterion(model(x, xyz_feats, edge_index, edge_attr, batch, pe, esm), y_true)
            loss.backward()
            return loss
        
        loss = model.criterion(y_pred, y_true)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model.optimizer.step(closure)

        epoch_loss_train += loss.item()
    epoch_loss_train_avg = epoch_loss_train / len(data_loader)
    # wandb.log({'total_train_loss':epoch_loss_train_avg})
    return epoch_loss_train_avg


def evaluate(use_esm, model, data_loader, valid=False, test=False):
    model.eval()
    epoch_loss = 0.0

    valid_pred = []
    valid_true = []

    pbar = tqdm(total=len(data_loader), desc='Evaluating')
    for data in data_loader:
        pbar.update(1)
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
    # if valid:
    #     wandb.log({'total_valid_loss':epoch_loss_avg})
    # if test:
    #     wandb.log({'total_test_loss':epoch_loss_avg})
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
    f1 = metrics.f1_score(binary_true, binary_pred, average='weighted')
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
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

def train_all(args, model, train_loader, test_loader, data_time):
    best_epoch = 0
    best_val_AUPRC = 0
    for epoch in range(args.epochs):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        epoch_loss_train_avg = train_one_epoch(args.use_esm, model, train_loader)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred = evaluate(args.use_esm, model, train_loader)
        result_train = analysis(train_true, train_pred, 0.85) # 0.85
        print(f"Train loss:{epoch_loss_train_avg}, Train binary acc:{result_train['binary_acc']}, Train AUC:{result_train['AUC']}, Train AUPRC:{result_train['AUPRC']}")

        print("========== Evaluate Test set ==========")
        epoch_loss_test_avg, test_true, test_pred = evaluate(args.use_esm, model, test_loader, test=True)
        result_test = analysis(test_true, test_pred, 0.85) # 0.85
        print(f"Test loss:{epoch_loss_test_avg}, Test binary acc:{result_test['binary_acc']}, Test precision:{result_test['precision']}, Test recall:{result_test['recall']}, Test f1:{result_test['f1']}, Test AUC:{result_test['AUC']}, Test AUPRC:{result_test['AUPRC']}, Test mcc:{result_test['mcc']}")

        if best_val_AUPRC < result_test['AUPRC']:
            best_epoch = epoch + 1
            best_val_AUPRC = result_test['AUPRC']
            torch.save(model.state_dict(), os.path.join(args.save_path, f'best_model_{data_time}.pth'))

        model.main_scheduler.step(result_test['AUPRC'])

    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val AUPRC: {best_val_AUPRC}")


def test_all(args, test_loader, data_time):
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

    model.load_state_dict(
        torch.load(os.path.join(args.save_path, f'best_model_{data_time}.pth'), map_location='cuda:0'))

    epoch_loss_test_avg, test_true, test_pred = evaluate(args.use_esm, model, test_loader)
    result_test = analysis(test_true, test_pred, 0.85) # 0.85

    return result_test

def train_with_no_cross_validation(args, trainset, testset, testset_PM, testset_P2RANK):
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    testPM_loader = DataLoader(testset_PM, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    testP2RANK_loader = DataLoader(testset_P2RANK, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    # 计算正负样本的权重
    all_labels = []
    for graph in train_loader:
        labels = graph.y.cpu().numpy()
        all_labels.extend(labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    channel = args.hidden_dim
    model = CrypKANet(
        channels=channel,
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

    model.get_optimizer_scheduler()

    data_time = get_time()
    train_all(args, model, train_loader, test_loader,data_time)
    test_result = test_all(args, test_loader, data_time)
    testPM_result = test_all(args, testPM_loader, data_time)
    testP2RANK_result = test_all(args, testP2RANK_loader, data_time)

    print("========== Result of Test ==========")
    args_dict = vars(args)
    for metric in sorted(test_result):
        print(f"{metric}: {test_result[metric]}")
    result_list = [args_dict,test_result]

    print("========== Result of Test PM ==========")
    for metric in sorted(testPM_result):
        print(f"{metric}: {testPM_result[metric]}")
    result_list.append(testPM_result)

    print("========== Result of Test P2RANK ==========")
    for metric in sorted(testP2RANK_result):
        print(f"{metric}: {testP2RANK_result[metric]}")
    result_list.append(testP2RANK_result)

    for dict in result_list:
        t = [f'{key}:{value}' for key, value in dict.items()]
        line = '   '.join(t)
        with open('result/result.txt','a') as f:
            f.write(data_time + '   '+ line+'\n')
    with open('result/result.txt','a') as f:
        f.write('\n')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = parse_args()
        
    print(args)
    set_seed(args.seed)
    
    os.makedirs(args.save_path, exist_ok=True)
    
    print("Creating Training set ...")
    start = time.time()
    train_set = ProDataset(
        pe_dim=args.pe_dim,
        dataset_name='train'
    )
    end = time.time()
    print("Creating datasets costs:", end-start)

    print("Creating Test set FULL ...")
    test_set = ProDataset(
        pe_dim=args.pe_dim,
        dataset_name="test_ALL",
    )

    print("Creating Test set PM ...")
    test_set_PM = ProDataset(
        pe_dim=args.pe_dim,
        dataset_name="test_PM",
    )

    print("Creating Test set P2RANK ...")
    test_set_P2RANK = ProDataset(
        pe_dim=args.pe_dim,
        dataset_name="test_P2RANK",
    )
    
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
        
    print("len of train:", len(train_set))
    print("len of test60: ", len(test_set))

    train_with_no_cross_validation(args, train_set, test_set, test_set_PM, test_set_P2RANK)
