import warnings
warnings.filterwarnings('ignore')
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.6"
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import time
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)
from scipy.stats import pearsonr, spearmanr
from dataset import load_embeddings, data_load, vocab_size
from modelTm import FinalModel
import argparse
from pathlib import Path



def parse_args():
    parser = argparse.ArgumentParser(description='Test Integrated Final Enzyme Model')
    
    tm_root = Path(__file__).resolve().parent.parent
    default_csv = str(tm_root / 'data' / 'Tm.csv')
    default_esm = str(tm_root / 'embedding')
    default_output_dir = str(tm_root / 'train_model')
    
    parser.add_argument('--csv_path', default=default_csv, type=str, help="Path to the dataset CSV file")
    parser.add_argument('--esm_path', default=default_esm, type=str, help="Directory containing ESM feature embeddings")
    parser.add_argument('--output_dir', default=default_output_dir, type=str, help="Directory to save final prediction results")
    
    parser.add_argument('--seq_max_len', default=1024, type=int, help="Maximum length for sequence padding")
    parser.add_argument('--batch_size', default=5, type=int, help="Batch size for testing")
    
    return parser.parse_args()


def seed_everything(seed=502):
   
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def performance(y_true_reg, y_pred_reg):
    
    
    y_true_reg = y_true_reg.flatten()
    y_pred_reg = y_pred_reg.flatten()
    r2 = r2_score(y_true_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    try:
        pearson = pearsonr(y_true_reg, y_pred_reg)[0]
    except:
        pearson = 0.0
    try:
        spearman = spearmanr(y_true_reg, y_pred_reg)[0]
    except:
        spearman = 0.0

    metrics = {
        "R2": round(r2,4),
        "Pearson": round(pearson,4),
        "Spearman": round(spearman,4),
        "RMSE": round(rmse,4),
        "MAE": round(mae,4)
    }

    return metrics

 


def final_test_safe(final_model, test_loader, device):
  
    final_model.eval()

    all_ids = []
    all_final = []
    all_cls_prob = []
    all_y1 = []
    all_y2 = []
    all_true = [] 

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Final Testing", mininterval=1.0)

        for batch in pbar:
            # unpack batch
            seq_inputs, reg_labels, cls_labels, esm_embeddings, esm_mask, weight, ids = batch

            seq_inputs = seq_inputs.to(device, non_blocking=True)
            if esm_embeddings is not None:
                esm_embeddings = esm_embeddings.to(device, non_blocking=True)
            if esm_mask is not None:
                esm_mask = esm_mask.to(device, non_blocking=True)

            
            final_pred, cls_prob, y1, y2 = final_model(seq_inputs, esm_embeddings, esm_mask)

       
            if isinstance(ids, torch.Tensor):
                ids = ids.view(-1).cpu().tolist()
            elif isinstance(ids, np.ndarray):
                ids = ids.flatten().tolist()

            
            final_pred = final_pred.detach().view(-1)
            cls_prob = cls_prob.detach().view(-1)
            y1 = y1.detach().view(-1)
            y2 = y2.detach().view(-1)
            true_vals = reg_labels.view(-1)

            
            all_ids.extend(ids)
            all_final.append(final_pred)
            all_cls_prob.append(cls_prob)
            all_y1.append(y1)
            all_y2.append(y2)
            all_true.append(true_vals)



    all_true = torch.cat(all_true).numpy()
    all_final = torch.cat(all_final).cpu().numpy()
    all_cls_prob = torch.cat(all_cls_prob).cpu().numpy()
    all_y1 = torch.cat(all_y1).cpu().numpy()
    all_y2 = torch.cat(all_y2).cpu().numpy()


    metrics_dict = performance(all_true, all_final)

    print(metrics_dict)

    
    df = pd.DataFrame({
        "ID": all_ids,
        "tm": all_true,
        "prediction": all_final,
        "cls_prob": all_cls_prob,
        "reg1_pred": all_y1,
        "reg2_pred": all_y2
    })

    return df, metrics_dict



# =========================
#        Main
# =========================
def main():
    seed_everything(502)
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 Preparing Data...")
    esm_features = load_embeddings(args.esm_path, max_length=args.seq_max_len)
    test_loader = data_load(
        csv_path=args.csv_path, 
        batch_size=args.batch_size, 
        split='Testing', 
        esm_loader=esm_features,
        seq_max_len=args.seq_max_len
    )


    print("📦 Initializing FinalTMModel and loading sub-model weights...")
    final_model = FinalModel(
        vocab_size=vocab_size,  
        device=device
    )


   
    # =======================
    #      Final Testing
    # =======================
    
    start_time = time.time()
    df, metrics = final_test_safe(final_model, test_loader, device)

    
    out_csv = os.path.join(args.output_dir, "EnzOracle_502.csv")
    df.to_csv(out_csv, index=False)
    print(f"✅ Final prediction saved to: {out_csv}")

    metrics_csv = os.path.join(args.output_dir, "final_metrics_log.csv")
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_csv, index=False)
    
    print(f"⏱️ Total testing time: {time.time() - start_time:.2f}s")



if __name__ == "__main__":
    main()
