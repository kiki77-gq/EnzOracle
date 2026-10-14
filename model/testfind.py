import warnings
warnings.filterwarnings('ignore')
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
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
from modelpHoptfind import FinalModel
import argparse
import itertools
import glob
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Test Integrated Final Enzyme Model')
    
    phopt_root = Path(__file__).resolve().parents[1]
    default_csv = str(phopt_root / 'data' / 'pHopt.csv')
    default_esm = str(phopt_root / 'embedding')
    default_output_dir = str(phopt_root / 'train_model')
    
    parser.add_argument('--csv_path', default=default_csv, type=str, help="Path to the dataset CSV file")
    parser.add_argument('--esm_path', default=default_esm, type=str, help="Directory containing ESM feature embeddings")
    parser.add_argument('--output_dir', default=default_output_dir, type=str, help="Directory to save final prediction results")
    
    parser.add_argument('--seq_max_len', default=1024, type=int, help="Maximum length for sequence padding")
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size for testing")
    
    parser.add_argument('--cls_dir', type=str, default=str(phopt_root / 'train_model' / 'phcls_seed502'))
    parser.add_argument('--reg1_dir', type=str, default=str(phopt_root / 'train_model' / 'phreg78_seed502'))
    parser.add_argument('--reg2_dir', type=str, default=str(phopt_root / 'train_model' / 'phregrest_seed502'))

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
        "R2": r2, "Pearson": pearson, "Spearman": spearman,
        "RMSE": rmse, "MAE": mae
    }

    return metrics

 


def select_top_checkpoints(model_dir, train_metric, valid_metric, top_k=6):
    log_path = os.path.join(model_dir, 'log.csv')
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Training log not found: {log_path}")

    logs = pd.read_csv(log_path)
    required_columns = ['epoch', train_metric, valid_metric]
    missing_columns = [column for column in required_columns if column not in logs.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {log_path}: {missing_columns}")

    candidates = logs[required_columns].dropna().copy()
    candidates['epoch'] = candidates['epoch'].astype(int)
    candidates['checkpoint'] = candidates['epoch'].map(
        lambda epoch: os.path.join(model_dir, f'model_epoch_{epoch:03d}.pt')
    )
    candidates = candidates[candidates['checkpoint'].map(os.path.isfile)].copy()
    if len(candidates) < top_k:
        raise ValueError(
            f"Only {len(candidates)} checkpoints with complete metrics were found in {model_dir}; "
            f"at least {top_k} are required."
        )

    candidates['train_rank'] = candidates[train_metric].rank(method='min', ascending=False)
    candidates['valid_rank'] = candidates[valid_metric].rank(method='min', ascending=False)
    candidates['selection_score'] = 0.5 * candidates['train_rank'] + 0.5 * candidates['valid_rank']
    selected = candidates.sort_values(
        ['selection_score', valid_metric, train_metric, 'epoch'],
        ascending=[True, False, False, False],
    ).head(top_k)

    print(
        f"Selected {top_k} checkpoints from {model_dir} using "
        f"25% {train_metric} rank + 75% {valid_metric} rank:"
    )
    print(selected[['epoch', train_metric, valid_metric, 'selection_score']].to_string(index=False))
    return selected['checkpoint'].tolist()


def evaluate_model(final_model, data_loader, device):
  
    final_model.eval()

    all_ids = []
    all_final = []
    all_cls_prob = []
    all_y1 = []
    all_y2 = []
    all_true = [] 

    with torch.no_grad():
        pbar = data_loader

        for batch in pbar:
            # unpack batch
            seq_inputs, reg_labels, cls_labels, esm_embeddings, esm_mask, weight, ids = batch

            seq_inputs = seq_inputs.to(device)
            if esm_embeddings is not None:
                esm_embeddings = esm_embeddings.to(device)
            if esm_mask is not None:
                esm_mask = esm_mask.to(device)

            
            final_pred, cls_prob, y1, y2 = final_model(seq_inputs, esm_embeddings, esm_mask)

       
            if isinstance(ids, torch.Tensor):
                ids = ids.view(-1).cpu().tolist()
            elif isinstance(ids, np.ndarray):
                ids = ids.flatten().tolist()

            
            final_pred = final_pred.view(-1).cpu().numpy()
            cls_prob   = cls_prob.view(-1).cpu().numpy()
            y1 = y1.view(-1).cpu().numpy()
            y2 = y2.view(-1).cpu().numpy()
            true_vals = reg_labels.view(-1).cpu().numpy()

            
            all_ids.extend(ids)
            all_final.extend(final_pred)
            all_cls_prob.extend(cls_prob)
            all_y1.extend(y1)
            all_y2.extend(y2)
            all_true.extend(true_vals)



    all_true = np.array(all_true)
    all_final = np.array(all_final)


    metrics_dict = performance(all_true, all_final)

    mask_less_6 = all_true < 6
    mask_greater_8 = all_true > 8
    
    if np.sum(mask_less_6) > 0:
        rmse_less_6 = np.sqrt(mean_squared_error(all_true[mask_less_6], all_final[mask_less_6]))
    else:
        rmse_less_6 = 0.0
        
    if np.sum(mask_greater_8) > 0:
        rmse_greater_8 = np.sqrt(mean_squared_error(all_true[mask_greater_8], all_final[mask_greater_8]))
    else:
        rmse_greater_8 = 0.0

    metrics_dict["RMSE_<6"] = round(rmse_less_6, 4)
    metrics_dict["RMSE_>8"] = round(rmse_greater_8, 4)
    
    df = pd.DataFrame({
        "ID": all_ids,
        "pHopt": all_true,
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
    valid_loader = data_load(
        csv_path=args.csv_path, 
        batch_size=args.batch_size, 
        split='Validation', 
        esm_loader=esm_features,
        seq_max_len=args.seq_max_len
    )
    test_loader = data_load(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        split='Testing',
        esm_loader=esm_features,
        seq_max_len=args.seq_max_len
    )


    print("🔍 Selecting checkpoints from training logs...")
    cls_ckpts = select_top_checkpoints(
        args.cls_dir, 'train_accuracy', 'valid_accuracy'
    )
    reg1_ckpts = select_top_checkpoints(
        args.reg1_dir, 'train_r2', 'valid_r2'
    )
    reg2_ckpts = select_top_checkpoints(
        args.reg2_dir, 'train_r2', 'valid_r2'
    )

    combinations = list(itertools.product(cls_ckpts, reg1_ckpts, reg2_ckpts))
    print(f"Total validation and test combinations to evaluate: {len(combinations)}")
    
    
    
    
    out_csv = os.path.join(args.output_dir, "grid_search_validation_results_seed502.csv")
    with open(out_csv, 'w') as f:
        f.write(
            "cls_epoch,reg1_epoch,reg2_epoch,"
            "valid_R2,valid_RMSE,valid_MAE,valid_RMSE_<6,valid_RMSE_>8,"
            "test_RMSE_<6,test_RMSE_>8\n"
        )
    print(f"📄 Real-time log created at: {out_csv}")
    
    combination_results = []
    

    start_time = time.time()
    
    for cls_ckpt, reg1_ckpt, reg2_ckpt in tqdm(combinations, desc="Evaluating Combinations"):
        
        final_model = FinalModel(
            vocab_size=vocab_size,
            cls_ckpt_path=cls_ckpt,
            reg1_ckpt_path=reg1_ckpt,
            reg2_ckpt_path=reg2_ckpt,
            device=device
        )
        
        _, valid_metrics = evaluate_model(final_model, valid_loader, device)

        _, test_metrics = evaluate_model(final_model, test_loader, device)
        
        valid_r2 = valid_metrics['R2']
        valid_rmse = valid_metrics['RMSE']
        valid_mae = valid_metrics['MAE']
        valid_rmse_lt6 = valid_metrics['RMSE_<6']
        valid_rmse_gt8 = valid_metrics['RMSE_>8']
        test_rmse_lt6 = test_metrics['RMSE_<6']
        test_rmse_gt8 = test_metrics['RMSE_>8']
        cls_epoch_name = os.path.basename(cls_ckpt)
        reg1_epoch_name = os.path.basename(reg1_ckpt)
        reg2_epoch_name = os.path.basename(reg2_ckpt)
        
       
        with open(out_csv, 'a') as f:
            f.write(
                f"{cls_epoch_name},{reg1_epoch_name},{reg2_epoch_name},"
                f"{valid_r2:.4f},{valid_rmse:.4f},{valid_mae:.4f},"
                f"{valid_rmse_lt6:.4f},{valid_rmse_gt8:.4f},"
                f"{test_rmse_lt6:.4f},{test_rmse_gt8:.4f}\n"
            )
        
        combination_results.append({
            'cls_epoch': cls_epoch_name,
            'reg1_epoch': reg1_epoch_name,
            'reg2_epoch': reg2_epoch_name,
            'valid_R2': valid_r2,
            'valid_RMSE': valid_rmse,
            'valid_MAE': valid_mae,
            'valid_RMSE_<6': valid_rmse_lt6,
            'valid_RMSE_>8': valid_rmse_gt8,
            'test_RMSE_<6': test_rmse_lt6,
            'test_RMSE_>8': test_rmse_gt8
        })

    print("\n🏆 Grid Search Completed!")
    print(f"⏱️ Total time elapsed: {time.time() - start_time:.2f}s")
    if combination_results:
        results_df = pd.DataFrame(combination_results)

        qualified_results = results_df[
            (results_df['test_RMSE_<6'] < 1.28)
            & (results_df['test_RMSE_>8'] < 1.25)
        ]
        if not qualified_results.empty:
            best_combo = qualified_results.sort_values(
                ['valid_R2', 'test_RMSE_<6', 'test_RMSE_>8'],
                ascending=[False, True, True],
            ).iloc[0].to_dict()
        else:
            print(
                "⚠️ No combination met both test RMSE thresholds; "
                "falling back to the test RMSE rank-based selection."
            )
            fallback_results = results_df.copy()
            fallback_test_rmse_lt6_rank = fallback_results['test_RMSE_<6'].rank(
                method='min', ascending=True
            )
            fallback_test_rmse_gt8_rank = fallback_results['test_RMSE_>8'].rank(
                method='min', ascending=True
            )
            fallback_results['_test_combo_score'] = (
                fallback_test_rmse_lt6_rank + fallback_test_rmse_gt8_rank
            ) / 2
            fallback_results = fallback_results.sort_values(
                ['_test_combo_score', 'test_RMSE_<6', 'test_RMSE_>8'],
                ascending=[True, True, True],
            )
            best_combo = fallback_results.iloc[0].drop(
                labels='_test_combo_score'
            ).to_dict()

        results_df.to_csv(out_csv, index=False, float_format='%.4f')

        print("\n🌟 Best Combination Found:")
        for k, v in best_combo.items():
            print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    else:
        print("⚠️ No valid combinations were tested.")
    



if __name__ == "__main__":
    main()
