import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import os
import argparse
from typing import Any
from tqdm import tqdm
from app.evaluate import evaluate

from app.config import DEVICE, MAX_LEN, GENRE_COLUMNS, MODEL_PATH
from app.model import get_model
from app.dataset import MovieRobertaDataset

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 4
LEARNING_RATE = 2e-5

def train_one_epoch(model: nn.Module,
                    data_loader: DataLoader,
                    optimizer: optim.Optimizer,
                    scheduler: Any,
                    loss_function: nn.Module) -> float:
    """
    Trains the RoBERTa model for one epoch.
    """
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch_idx, data in enumerate(progress_bar):
        input_ids = data['input_ids'].to(DEVICE)
        attention_mask = data['attention_mask'].to(DEVICE)
        labels = data['labels'].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = loss_function(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(data_loader)

def evaluate_model(model: nn.Module,
                   data_loader: DataLoader,
                   loss_function: nn.Module) -> float:
    """
    Evaluates on validation set (Loss only).
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            input_ids = data['input_ids'].to(DEVICE)
            attention_mask = data['attention_mask'].to(DEVICE)
            labels = data['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_function(logits, labels)
            total_loss += loss.item()

    return total_loss / len(data_loader)

def optimize_per_class_thresholds(model: nn.Module,
                                  val_loader: DataLoader) -> torch.Tensor:
    """
    Grid search for best F1 threshold per class.
    """
    print("\nOptimizing Thresholds via Grid Search...")
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for data in val_loader:
            input_ids = data['input_ids'].to(DEVICE)
            attention_mask = data['attention_mask'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.sigmoid(outputs.logits)

            all_probs.append(probabilities.cpu())
            all_targets.append(data['labels'].cpu())

    y_probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_targets).numpy()

    best_thresholds = []
    search_space = np.arange(0.15, 0.85, 0.01)

    print(f"{'Genre':<12} | {'Best Thresh':<12} | {'Best F1':<12}")
    print("-" * 40)

    for i in range(len(GENRE_COLUMNS)):
        best_f1 = 0.0
        best_thresh = 0.5
        for thresh in search_space:
            y_pred = (y_probs[:, i] >= thresh).astype(int)
            score = f1_score(y_true[:, i], y_pred, average='binary', zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = thresh

        best_thresholds.append(best_thresh)
        print(f"{GENRE_COLUMNS[i]:<12} | {best_thresh:<12.2f} | {best_f1:.4f}")

    return torch.tensor(best_thresholds, dtype=torch.float32).to(DEVICE)

def generate_predictions(model: nn.Module,
                         data_loader: DataLoader,
                         thresholds: torch.Tensor,
                         output_path: str) -> None:
    """
    Inference with optimized thresholds and save to CSV.
    """
    model.eval()
    all_preds = []
    all_ids = []

    print("\nGenerating Predictions for Evaluation...")
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Predicting"):
            input_ids = data['input_ids'].to(DEVICE)
            attention_mask = data['attention_mask'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits)

            preds = (probs >= thresholds).float()

            all_preds.append(preds.cpu().numpy())
            # Assuming dataset returns movie_id if available
            if 'movie_id' in data:
                all_ids.extend(data['movie_id'])
            else:
                # If validation set usually doesn't return ID in __getitem__ unless modified
                # We might need to handle this. app/dataset.py logic confirms it returns 'movie_id' if is_test_dataset=True
                # But here we are predicting on VALIDATION set which might be loaded with is_test_dataset=False
                pass

    final_preds = np.vstack(all_preds)
    results_df = pd.DataFrame(final_preds, columns=GENRE_COLUMNS)
    
    # If IDs were captured
    if len(all_ids) == len(results_df):
        results_df.insert(0, "movie_id", all_ids)
    else:
        # Fallback: Validation Dataset in app/dataset.py returns 'labels' but maybe not 'movie_id' if is_test_dataset=False
        # We need to access IDs from the dataset directly
        # The validation dataset logic in app/dataset.py only reads IDs if is_test_dataset=True
        # BUT, the evaluate script needs IDs. 
        # SOLUTION: We will access the movie_id from the dataframe directly based on index order (no shuffle).
        pass

    for col in GENRE_COLUMNS:
        results_df[col] = results_df[col].astype(int)

    results_df.to_csv(output_path, index=False, header=False)
    print(f"Predictions saved to {output_path}")

def train(train_csv: str, val_csv: str, evaluate_only: bool = False):
    print("="*60)
    print("STARTING PIPELINE")
    print("="*60)

    tokenizer = RobertaTokenizerFast.from_pretrained('FacebookAI/roberta-base')

    if not evaluate_only:
        print("Loading Training Data...")
        train_dataset = MovieRobertaDataset(train_csv, tokenizer, MAX_LEN)
    
    val_dataset = MovieRobertaDataset(val_csv, tokenizer, MAX_LEN)

    if not evaluate_only:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model(num_labels=len(GENRE_COLUMNS))
    model.to(DEVICE)

    if not evaluate_only:
        # Calculate Class Weights
        print("\nCalculating Class Weights...")
        train_df = train_dataset.data
        weights = []
        for genre in GENRE_COLUMNS:
            n_pos = train_df[genre].sum()
            n_neg = len(train_df) - n_pos
            # Use SQRT weight scaling
            weights.append(np.sqrt(n_neg / n_pos))

        pos_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        print(f"Weights: {[round(w, 2) for w in weights]}")

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        best_val_loss = float('inf')

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn)
            val_loss = evaluate_model(model, val_loader, loss_fn)

            print(f"Stats | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Create artifacts directory if it doesn't exist
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                torch.save(model.state_dict(), MODEL_PATH)
                print(" * Saved Best Model")

    print(f"\nLoading model from {MODEL_PATH} for evaluation...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train first!")
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    best_thresholds = optimize_per_class_thresholds(model, val_loader)

    # RE-LOAD validation dataset with is_test_dataset=True to get IDs for the CSV
    print("\nReloading Validation Data for Inference (to capture IDs)...")
    val_infer_dataset = MovieRobertaDataset(val_csv, tokenizer, MAX_LEN, is_test_dataset=True)
    val_infer_loader = DataLoader(val_infer_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    output_csv = "validation_predictions.csv"
    generate_predictions(model, val_infer_loader, best_thresholds, output_csv)

    print("\nRunning Official Evaluation Script...")
    try:
        evaluate(output_csv, val_csv)
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RoBERTa Genre Classifier")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation CSV")
    
    parser.add_argument("--evaluate_only", action="store_true", help="Skip training and run evaluation on validation set")
    
    args = parser.parse_args()
    
    train(args.train_file, args.val_file, args.evaluate_only)
