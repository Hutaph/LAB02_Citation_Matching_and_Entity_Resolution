"""
Modeling Utilities

This module provides utility functions for model training, evaluation, and prediction
for the citation matching task. Functions are designed to be called from notebooks
to keep the notebook code clean and maintainable.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, PredefinedSplit

# Optional progress bar
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False


# Feature columns used for training
FEATURE_COLS = [
    'levenshtein',
    'year_match', 'year_diff',
    'source_year', 'cand_year',
    'token_overlap', 'token_overlap_ratio',
    'char_ngram_3', 'char_ngram_4', 'char_ngram_5',
    'author_overlap',
    'author_lastname_match',
    'author_firstname_match'
]


def setup_paths(root: Optional[Path] = None) -> Dict[str, Path]:
    """
    Setup and return paths for modeling pipeline.
    
    Args:
        root: Root directory of the project. If None, resolves from current directory.
        
    Returns:
        Dictionary containing:
        - 'root': Project root directory
        - 'split_dir': Directory containing split JSONL files
        - 'model_dir': Directory for saved models
        - 'model_path': Path to saved model file
        - 'processed_dir': Directory containing processed papers
    """
    if root is None:
        root = Path("..").resolve()
    else:
        root = Path(root).resolve()
    
    split_dir = root / "split"
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "citation_matcher_rf.pkl"
    processed_dir = root / "23120334"
    
    return {
        'root': root,
        'split_dir': split_dir,
        'model_dir': model_dir,
        'model_path': model_path,
        'processed_dir': processed_dir
    }


def load_split_data(split_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load pre-split data from train/val/test JSONL files.
    
    Args:
        split_dir: Directory containing split JSONL files
        
    Returns:
        Tuple of (train_df, val_df, test_df)
        
    Raises:
        SystemExit: If any split file is missing
    """
    train_path = split_dir / "train.jsonl"
    val_path = split_dir / "val.jsonl"
    test_path = split_dir / "test.jsonl"
    
    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise SystemExit(f"Missing {p}; run match_and_fe.ipynb split step first.")
    
    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    
    train_df["partition"] = "train"
    val_df["partition"] = "val"
    test_df["partition"] = "test"
    
    return train_df, val_df, test_df


def sample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_size: int = 500000,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Sample training data while keeping all positive samples.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sample_size: Maximum number of samples to keep
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (sampled_X_train, sampled_y_train)
    """
    if len(X_train) <= sample_size:
        return X_train, y_train
    
    np.random.seed(random_seed)
    
    # Keep all positive samples
    y_train_values = y_train.values if hasattr(y_train, 'values') else y_train
    pos_mask = y_train_values == 1
    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(~pos_mask)[0]
    
    # Sample negatives
    n_neg_samples = min(sample_size - len(pos_indices), len(neg_indices))
    sampled_neg_indices = np.random.choice(neg_indices, size=n_neg_samples, replace=False)
    
    # Combine indices
    sampled_indices = np.concatenate([pos_indices, sampled_neg_indices])
    np.random.shuffle(sampled_indices)
    
    # Sample from DataFrame
    sampled_X = X_train.iloc[sampled_indices].reset_index(drop=True)
    sampled_y = y_train.iloc[sampled_indices].reset_index(drop=True)
    
    print(f"Sampled training set: {len(sampled_X)} rows "
          f"({sampled_y.sum()} positive, {len(sampled_y) - sampled_y.sum()} negative)")
    
    return sampled_X, sampled_y


def prepare_data_for_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    enable_sampling: bool = True,
    sample_size: int = 500000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, PredefinedSplit]:
    """
    Prepare data for model training with optional sampling.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature column names
        enable_sampling: Whether to sample training data if too large
        sample_size: Maximum sample size for training data
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, pds)
        where pds is PredefinedSplit for cross-validation
    """
    X_train, y_train = train_df[feature_cols], train_df['label']
    X_val, y_val = val_df[feature_cols], val_df['label']
    X_test, y_test = test_df[feature_cols], test_df['label']
    
    # Sample training data if enabled and too large
    if enable_sampling and len(X_train) > sample_size:
        print(f"Training set is large ({len(X_train)} rows). Sampling {sample_size} rows...")
        X_train, y_train = sample_training_data(X_train, y_train, sample_size)
    
    # Convert to numpy arrays
    X_train = X_train.values if hasattr(X_train, 'values') else X_train
    y_train = y_train.values if hasattr(y_train, 'values') else y_train
    X_val = X_val.values if hasattr(X_val, 'values') else X_val
    y_val = y_val.values if hasattr(y_val, 'values') else y_val
    X_test = X_test.values if hasattr(X_test, 'values') else X_test
    y_test = y_test.values if hasattr(y_test, 'values') else y_test
    
    # Combine Train and Validation sets for PredefinedSplit
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    
    # Create PredefinedSplit
    split_index = [-1] * len(X_train) + [0] * len(X_val)
    pds = PredefinedSplit(test_fold=split_index)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, pds


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pds: PredefinedSplit,
    param_grid: Optional[Dict] = None,
    random_state: int = 42,
    n_jobs: int = 1,
    verbose: int = 1
) -> Tuple[RandomForestClassifier, GridSearchCV]:
    """
    Train RandomForest model with GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        pds: PredefinedSplit for cross-validation
        param_grid: Hyperparameter grid (default: predefined grid)
        random_state: Random state for reproducibility
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        
    Returns:
        Tuple of (best_model, grid_search)
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [200, 250],
            'max_depth': [None, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced', 'balanced_subsample']
        }
    
    # Calculate total combinations
    total_combinations = 1
    for key, values in param_grid.items():
        total_combinations *= len(values)
    
    print(f"Total parameter combinations to test: {total_combinations}")
    
    # Combine train and val for PredefinedSplit
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    
    # Initialize Grid Search
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=pds,
        scoring='f1',
        verbose=verbose,
        n_jobs=n_jobs,
        return_train_score=True
    )
    
    print(f"\nStarting Grid Search ({total_combinations} combinations)...")
    print("This will test all parameter combinations exhaustively.")
    grid_search.fit(X_combined, y_combined)
    
    best_model = grid_search.best_estimator_
    
    print("\n" + "="*50)
    print("Tuning Complete.")
    print("="*50)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Val Score:  {grid_search.best_score_:.4f}")
    print(f"Total combinations tested: {total_combinations}")
    
    return best_model, grid_search


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
) -> Dict:
    """
    Evaluate model on test and optionally validation sets.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        X_val: Optional validation features
        y_val: Optional validation labels
        
    Returns:
        Dictionary containing predictions, probabilities, and metrics
    """
    results = {}
    
    # Test set predictions
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    results['test'] = {
        'y_pred': y_pred_test,
        'y_prob': y_prob_test,
        'y_true': y_test,
        'report': classification_report(y_test, y_pred_test, labels=[0, 1], output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test)
    }
    
    # Validation set predictions (if provided)
    if X_val is not None and y_val is not None:
        y_pred_val = model.predict(X_val)
        results['val'] = {
            'y_pred': y_pred_val,
            'y_prob': model.predict_proba(X_val)[:, 1],
            'y_true': y_val,
            'report': classification_report(y_val, y_pred_val, labels=[0, 1], output_dict=True),
            'confusion_matrix': confusion_matrix(y_val, y_pred_val)
        }
    
    return results


def calculate_mrr(ranked_list: List[str], true_id: str) -> float:
    """
    Calculate Reciprocal Rank: 1/rank if true_id in top 5, else 0.
    
    Args:
        ranked_list: List of candidate IDs ranked by score
        true_id: True candidate ID
        
    Returns:
        Reciprocal rank (0.0 if not in top 5)
    """
    for i, cand_id in enumerate(ranked_list[:5]):
        if cand_id == true_id:
            return 1.0 / (i + 1)
    return 0.0


def compute_mrr_scores(
    test_df: pd.DataFrame,
    model: RandomForestClassifier,
    feature_cols: List[str]
) -> Tuple[List[float], int]:
    """
    Compute MRR scores for test set.
    
    Args:
        test_df: Test DataFrame with features and labels
        model: Trained model
        feature_cols: List of feature column names
        
    Returns:
        Tuple of (mrr_scores list, number of queries)
    """
    if len(test_df) == 0:
        return [], 0
    
    # Get predictions
    X_test = test_df[feature_cols].values
    test_df = test_df.copy()
    test_df['score'] = model.predict_proba(X_test)[:, 1]
    
    mrr_scores = []
    
    # Group by Paper ID and Citation Key
    for paper_id, paper_group in test_df.groupby('paper_id'):
        for bib_key, bib_group in paper_group.groupby('bib_key'):
            # Rank candidates
            ranked = bib_group.sort_values('score', ascending=False)
            top_5 = ranked['cand_id'].head(5).tolist()
            
            # Get ground truth
            true_row = bib_group[bib_group['label'] == 1]
            if not true_row.empty:
                true_id = true_row.iloc[0]['cand_id']
                rr = calculate_mrr(top_5, true_id)
                mrr_scores.append(rr)
    
    return mrr_scores, len(mrr_scores)


def update_pred_json_files(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model: RandomForestClassifier,
    feature_cols: List[str],
    processed_dir: Path,
    show_progress: bool = True
) -> Dict:
    """
    Update pred.json files with top 5 predictions for all papers.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        model: Trained model
        feature_cols: List of feature column names
        processed_dir: Directory containing processed papers
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary with statistics
    """
    # Combine all data
    full_eval_df = pd.concat([test_df, val_df, train_df], ignore_index=True)
    X_full = full_eval_df[feature_cols].values
    full_eval_df = full_eval_df.copy()
    full_eval_df['score'] = model.predict_proba(X_full)[:, 1]
    
    stats = {
        'total_papers': 0,
        'updated_papers': 0,
        'missing_files': 0,
        'total_bib_keys': 0,
        'test_queries': 0,
        'mrr_scores': []
    }
    
    # Process by paper
    papers = list(full_eval_df.groupby('paper_id'))
    if show_progress and TQDM_AVAILABLE:
        papers = tqdm(papers, desc="Updating pred.json")
    
    for paper_id, paper_group in papers:
        stats['total_papers'] += 1
        save_path = processed_dir / paper_id / "pred.json"
        
        # Read existing pred.json or create new structure
        if save_path.exists():
            try:
                with open(save_path, 'r', encoding='utf-8') as f:
                    pred_json = json.load(f)
            except:
                pred_json = {"partition": "unknown", "groundtruth": {}, "prediction": {}}
        else:
            partition = paper_group.iloc[0].get('partition', 'unknown') if 'partition' in paper_group.columns else 'unknown'
            pred_json = {"partition": partition, "groundtruth": {}, "prediction": {}}
            save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update predictions for each bib_key
        for bib_key, bib_group in paper_group.groupby('bib_key'):
            stats['total_bib_keys'] += 1
            
            # Get top 5 candidates by score
            ranked = bib_group.sort_values('score', ascending=False)
            top_5 = ranked['cand_id'].head(5).tolist()
            
            # Update prediction
            pred_json["prediction"][bib_key] = top_5
            
            # Extract groundtruth from data (ONLY for evaluation, NOT used in prediction)
            if bib_key not in pred_json["groundtruth"]:
                true_row = bib_group[bib_group['label'] == 1]
                if not true_row.empty:
                    pred_json["groundtruth"][bib_key] = true_row.iloc[0]['cand_id']
            
            # Calculate MRR for test set (evaluation only)
            if pred_json.get("partition") == "test" and bib_key in pred_json["groundtruth"]:
                true_id = pred_json["groundtruth"][bib_key]
                rr = calculate_mrr(top_5, true_id)
                stats['mrr_scores'].append(rr)
                stats['test_queries'] += 1
        
        # Write updated file
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(pred_json, f, indent=4, ensure_ascii=False)
        stats['updated_papers'] += 1
    
    return stats


def get_feature_importance(model: RandomForestClassifier, feature_cols: List[str]) -> pd.DataFrame:
    """
    Get feature importance from trained model.
    
    Args:
        model: Trained RandomForest model
        feature_cols: List of feature column names
        
    Returns:
        DataFrame with features and their importance scores
    """
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    })
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    return feature_imp_df

