from sklearn.model_selection import StratifiedKFold
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score, matthews_corrcoef, precision_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import statistics
from model import DFL_MHC

all_train_acc = []
all_train_sn = []
all_train_sp = []
all_train_mcc = []

all_test_acc = []



def train_model(model, train_loader, val_loader, scheduler, train_idx, criterion, optimizer, device, save_path):

    best_acc = 0
    best_sn = 0
    best_sp = 0
    best_MCC = 0
    best_val_f1 = 0.0

    epochs_no_improve = 0
    patience = 100

    for epoch in tqdm(range(5000)):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        print(f"Epoch {epoch + 1}: Train Loss = {avg_loss:.8f}")

        # val
        model.eval()
        all_preds, all_labels = [], []
        misclassified_indices = []
        val_index_counter = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)

                for i in range(len(preds)):
                    if preds[i] != yb[i]:
                        misclassified_indices.append(val_index_counter + i)
                val_index_counter += len(preds)

                all_preds.append(preds.cpu())
                all_labels.append(yb.cpu())

        all_preds = torch.cat(all_preds).cpu().numpy().flatten()
        all_labels = torch.cat(all_labels).cpu().numpy().flatten()
        acc = round(accuracy_score(all_labels, all_preds), 4)
        sn = round(recall_score(all_labels, all_preds, average='macro'), 4)
        sp = round(precision_score(all_labels, all_preds, average='macro'), 4)
        mcc = round(matthews_corrcoef(all_labels, all_preds), 4)
        f1 = f1_score(all_labels, all_preds, average='macro')

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch + 1}: Val F1 improved to {f1:.4f}, ACC={acc:.4f}, SN={sn:.4f}, SP={sp:.4f}, MCC={mcc:.4f}")
            epochs_no_improve = 0
        else:
            print(f"Epoch {epoch + 1}: Val F1 = {f1:.4f}, ACC={acc:.4f}, SN={sn:.4f}, SP={sp:.4f}, MCC={mcc:.4f}")
            epochs_no_improve += 1

        if acc > best_acc:
            best_acc = acc
            best_sn = sn
            best_sp = sp
            best_MCC = mcc
        print(f" Best: ACC={best_acc:.4f}, SN={best_sp:.4f}, SP={best_sn:.4f}, MCC={best_MCC:.4f}")

        scheduler.step()

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1} — 超过 {patience} 个 epoch 验证集 F1 无提升")
            break

    all_train_acc.append(best_acc)
    all_train_sn.append(best_sn)
    all_train_sp.append(best_sp)
    all_train_mcc.append(best_MCC)

    return best_val_f1

def test_model(model, test_loader, test_idx,device):

    best_acc = 0
    best_sn = 0
    best_sp = 0
    best_MCC = 0

    model.eval()
    all_preds, all_labels = [], []
    misclassified_indices = []
    test_index_counter = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)

            for i in range(len(preds)):
                if preds[i] != yb[i]:
                    misclassified_indices.append(test_index_counter + i)
            test_index_counter += len(preds)

            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

    all_preds = torch.cat(all_preds).cpu().numpy().flatten()
    all_labels = torch.cat(all_labels).cpu().numpy().flatten()
    acc = round(accuracy_score(all_labels, all_preds), 4)
    sn = round(recall_score(all_labels, all_preds, average='macro'), 4)
    sp = round(precision_score(all_labels, all_preds, average='macro'), 4)
    mcc = round(matthews_corrcoef(all_labels, all_preds), 4)
    f1 = f1_score(all_labels, all_preds, average='macro')

    if acc > best_acc:
        best_acc = acc
        best_sn = sn
        best_sp = sp
        best_MCC = mcc
    print(
        f"Test Best: ACC={best_acc:.4f}, SN={best_sp:.4f}, SP={best_sn:.4f}, MCC={best_MCC:.4f}")

    all_test_acc.append(best_acc)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = 'pca_data'
    X_train = np.load(f"{save_dir}/best_train_X.npy")
    y_train = np.load(f"{save_dir}/best_train_y.npy")
    train_idx = np.load(f"{save_dir}/best_train_idx.npy")

    X_test = np.load(f"{save_dir}/best_test_X.npy")
    y_test = np.load(f"{save_dir}/best_test_y.npy")
    test_idx = np.load(f"{save_dir}/best_test_idx.npy")

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    test_loader = DataLoader(test_dataset, batch_size=32)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    global_best_f1 = 0
    global_best_model_path = None

    for fold, (train_split_idx, val_split_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n=== Fold {fold} ===")

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        train_set = TensorDataset(
            X_train_tensor[train_split_idx],
            y_train_tensor[train_split_idx]
        )

        val_set = TensorDataset(
            X_train_tensor[val_split_idx],
            y_train_tensor[val_split_idx]
        )

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32)

        val_sub_idx = train_idx[val_split_idx]

        model = DFL_MHC().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=0.0001,
            betas=(0.9, 0.999),
            weight_decay=1e-8
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1000, gamma=0.1
        )

        save_path = f"best_model_fold_{fold}.pt"

        # train
        best_f1 = train_model(model,
                    train_loader,
                    val_loader,
                    scheduler,
                    val_sub_idx,
                    criterion,
                    optimizer,
                    device,
                    save_path)

        # test
        model.load_state_dict(torch.load(save_path))

        if best_f1 > global_best_f1:
            global_best_f1 = best_f1
            global_best_model_path = save_path

    print(f"Train Result: ACC:{round(statistics.mean(all_train_acc), 4)}, SN:{round(statistics.mean(all_train_sn), 4)},"
          f"SP:{round(statistics.mean(all_train_sp), 4)}, MCC:{round(statistics.mean(all_train_mcc), 4)}")

    final_model = DFL_MHC().to(device)
    final_model.load_state_dict(torch.load(global_best_model_path))

    test_model(
        final_model,
        test_loader,
        test_idx,
        device
    )

if __name__ == '__main__':
    main()