import warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.6"
import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from modelcls import *
from tqdm import tqdm
import time
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from dataset import load_embeddings, cls_data_load_seq, vocab_size





def seed_everything(seed=42):


    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def performance(y_true_cls, y_pred_cls_prob, threshold=0.5):

   
    if isinstance(y_pred_cls_prob, torch.Tensor):
        y_pred_cls_prob = y_pred_cls_prob.detach().cpu().numpy()
    if isinstance(y_true_cls, torch.Tensor):
        y_true_cls = y_true_cls.detach().cpu().numpy()

   
    y_pred_cls = (y_pred_cls_prob >= threshold).astype(int) 

    accuracy = accuracy_score(y_true_cls, y_pred_cls)
    precision = precision_score(y_true_cls, y_pred_cls, zero_division=0)
    recall = recall_score(y_true_cls, y_pred_cls, zero_division=0)
    f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)

    metrics = {
        "Accuracy": round(accuracy, 4),
        "F1": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4)
    }

    return metrics








def write_logfile(test_loss, test_metrics, logfile):
   
    columns = [
        "test_loss", "test_accuracy", "test_f1", "test_precision", "test_recall"
    ]

    values = [
        test_loss,
        test_metrics.get("Accuracy", np.nan),
        test_metrics.get("F1", np.nan),
        test_metrics.get("Precision", np.nan),
        test_metrics.get("Recall", np.nan)
    ]

    df = pd.DataFrame([values], columns=columns)

    if not os.path.exists(logfile):
        df.to_csv(logfile, index=False, float_format="%.4f")
    else:
        df.to_csv(logfile, mode='a', header=False, index=False, float_format="%.4f")


def log_epoch_results(start_time, test_loss, test_metrics):
    
    elapsed = time.time() - start_time
    print(
        f"Test Loss: {test_loss:.4f}, Accuracy: {test_metrics.get('Accuracy',0):.4f} | "
        f"Time: {elapsed:.2f}s"
    )




def test_one_epoch(model, dataloader, device, cls_loss_func, use_amp=False, find_best_thresh=False):
 

    model.eval() 
    total_loss_samples = torch.zeros((), device=device)
    total_samples = 0

    all_cls_true = []
    all_cls_pred_prob = []
    all_ids = []


    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing", mininterval=1.0)

        for batch_idx, batch in enumerate(pbar):
            seq_inputs, reg_labels, cls_labels, esm_embeddings, esm_mask, weight, ids = batch
            seq_inputs = seq_inputs.to(device, non_blocking=True)
            cls_labels = cls_labels.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True).squeeze()
            if esm_embeddings is not None:
                esm_embeddings = esm_embeddings.to(device, non_blocking=True)
            if esm_mask is not None:
                esm_mask = esm_mask.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
               
                cls_probs, fusion, att = model(seq_inputs, esm_embeddings, esm_mask)  

            
                loss = cls_loss_func(cls_probs, cls_labels)  


            batch_size = seq_inputs.size(0)
            total_loss_samples += loss.detach() * batch_size
            total_samples += batch_size

            
            all_ids.extend(ids)
            all_cls_true.append(cls_labels.detach())
            all_cls_pred_prob.append(cls_probs.detach())


            if batch_idx % 20 == 0:
                pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

   
    avg_loss = (total_loss_samples / len(dataloader.dataset)).item()

 
    all_cls_true = torch.cat(all_cls_true).cpu().numpy()
    all_cls_pred_prob = torch.cat(all_cls_pred_prob).cpu().numpy()


       
    metrics_dict = performance(all_cls_true, all_cls_pred_prob)

  
  

    return avg_loss, metrics_dict, all_cls_true, all_cls_pred_prob, all_ids



def parse_args():
    parser = argparse.ArgumentParser(description='Test Enzyme Classification Model')
    

    tm_root = Path(__file__).resolve().parents[2]
    default_csv = str(tm_root / 'data' / 'Tm.csv')
    default_esm = str(tm_root / 'embedding')
    default_output_dir = str(tm_root / 'train_model' / 'tmcls_seed1214')
    default_model_path = os.path.join(default_output_dir, 'model_best.pt')
    
    parser.add_argument('--csv_path', default=default_csv, type=str, help="Path to the dataset CSV file")
    parser.add_argument('--esm_path', default=default_esm, type=str, help="Directory containing ESM feature embeddings")
    parser.add_argument('--model_path', default=default_model_path, type=str, help="Path to the best model checkpoint")
    parser.add_argument('--output_dir', default=default_output_dir, type=str, help="Directory to save test results and logs")
    
    parser.add_argument('--seq_max_len', default=1024, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    
    return parser.parse_args()



def main():


    seed_everything(42)
    args = parse_args()
    esm_features = load_embeddings(args.esm_path, max_length=args.seq_max_len)
    test_loader = cls_data_load_seq(csv_path=args.csv_path, split='Testing', batch_size=args.batch_size, esm_loader=esm_features, seq_max_len=args.seq_max_len)
    cls_loss_func = nn.BCELoss()

    model = Mymodel_tm(vocab_size=vocab_size).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print("✅ Model weights loaded successfully.")
    log_path = os.path.join(args.output_dir, "test_0107.csv")


    start_time = time.time()


   
    test_loss, test_metrics, cls_true, cls_prob, ids = test_one_epoch(
        model, test_loader, device, cls_loss_func)
    
    if cls_prob.ndim > 1 and cls_prob.shape[1] == 1:
        cls_prob = cls_prob.squeeze(1)

    cls_true = np.array(cls_true).flatten()
    cls_prob = np.array(cls_prob).flatten()  

   
    save_df = pd.DataFrame({
        "ID": ids,
        "cls_true": cls_true,
        "cls_prob": cls_prob
    })
    pred_csv_path = os.path.join(args.output_dir, "test_pred_results0107.csv")
    save_df.to_csv(pred_csv_path, index=False)
    print("✅ Saved prediction results to CSV.")


   
    write_logfile(test_loss, test_metrics, log_path)
    log_epoch_results(start_time, test_loss, test_metrics)



if __name__ == "__main__":
    main()




