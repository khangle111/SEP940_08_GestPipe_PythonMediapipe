import os
import pickle
from pathlib import Path

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# === Config ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(BASE_DIR, "gesture_motion_dataset_realistic.csv")
RESULTS_DIR = Path(BASE_DIR) / "training_results"
MODELS_DIR = Path(BASE_DIR) / "models"

LEFT_COLS = [f"left_finger_state_{i}" for i in range(5)]
RIGHT_COLS = [f"right_finger_state_{i}" for i in range(5)]
MOTION_COLS = ["main_axis_x", "main_axis_y", "delta_x", "delta_y"]

DELTA_WEIGHT = 15.0  # Increased from 5.0 to emphasize motion direction
MIN_DELTA_MAG = 0.001  # Lowered to preserve static gesture data (was 0.05)

COARSE_C_VALUES = [0.1, 1, 10]  # Further reduced to prevent system overload  
COARSE_GAMMA_VALUES = [0.01, 0.1, "auto"]  # Further reduced to prevent system overload
FINE_MULTIPLIERS = [1.0, 2.0]  # Further reduced to prevent system overload

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PKL = str(MODELS_DIR / "motion_svm_model.pkl")
SCALER_PKL = str(MODELS_DIR / "motion_scaler.pkl")
STATIC_DYNAMIC_PKL = str(MODELS_DIR / "static_dynamic_classifier.pkl")


TEST_FRACTION = 0.25
PREDICT_MIN_PROB = 0.6

# Static/Dynamic gestures classification - Auto-detection with manual override
STATIC_THRESHOLD = 0.01  # Delta threshold for static gestures
MANUAL_STATIC_OVERRIDE = None  # ['home', 'end'] if want to override
MANUAL_DYNAMIC_OVERRIDE = None  # ['zoom_in', ...] if want to override


# === Data utilities ===
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing dataset: {path}")
    df = pd.read_csv(path)
    
    # Check required columns - handle both old and new dataset formats
    required_base = LEFT_COLS + RIGHT_COLS + MOTION_COLS + ["pose_label"]
    missing_base = [col for col in required_base if col not in df.columns]
    if missing_base:
        raise ValueError(f"Dataset missing required columns: {missing_base}")
    
    # Add base_instance_id if not present (for new datasets)
    if "base_instance_id" not in df.columns:
        df["base_instance_id"] = df["instance_id"] if "instance_id" in df.columns else range(len(df))
        print(f"[INFO] Added base_instance_id column to dataset")
    
    # Add instance_id if not present
    if "instance_id" not in df.columns:
        df["instance_id"] = range(len(df))
        print(f"[INFO] Added instance_id column to dataset")
    
    return df


def prepare_features(df: pd.DataFrame):
    df = df.copy()

    # enforce numeric types
    for col in LEFT_COLS + RIGHT_COLS:
        df[col] = df[col].fillna(0).astype(int)

    delta_x_series = df["delta_x"].astype(float)
    delta_y_series = df["delta_y"].astype(float)
    axis_x = (delta_x_series.abs() >= delta_y_series.abs()).astype(int)
    axis_y = 1 - axis_x
    df["main_axis_x"] = axis_x.astype(float)
    df["main_axis_y"] = axis_y.astype(float)
    df.loc[axis_x == 1, "delta_x"] = delta_x_series[axis_x == 1]
    df.loc[axis_x == 1, "delta_y"] = 0.0
    df.loc[axis_x == 0, "delta_y"] = delta_y_series[axis_x == 0]
    df.loc[axis_x == 0, "delta_x"] = 0.0

    df["delta_mag"] = np.sqrt(df["delta_x"] ** 2 + df["delta_y"] ** 2)
    before = len(df)
    df = df[df["delta_mag"] >= MIN_DELTA_MAG].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[INFO] Dropped {dropped} samples with delta_mag < {MIN_DELTA_MAG}")
    df = df.drop(columns=["delta_mag"])

    df.loc[:, "delta_x"] = df["delta_x"] * DELTA_WEIGHT
    df.loc[:, "delta_y"] = df["delta_y"] * DELTA_WEIGHT
    
    # Add explicit direction features to help distinguish similar finger states
    df["motion_left"] = (df["delta_x"] < 0).astype(float) * DELTA_WEIGHT
    df["motion_right"] = (df["delta_x"] > 0).astype(float) * DELTA_WEIGHT
    df["motion_up"] = (df["delta_y"] < 0).astype(float) * DELTA_WEIGHT  
    df["motion_down"] = (df["delta_y"] > 0).astype(float) * DELTA_WEIGHT

    finger_feats = df[LEFT_COLS + RIGHT_COLS].values.astype(float)
    motion_feats = df[MOTION_COLS + ["motion_left", "motion_right", "motion_up", "motion_down"]].values.astype(float)

    scaler = StandardScaler()
    motion_scaled = scaler.fit_transform(motion_feats)

    X = np.hstack([finger_feats, motion_scaled])
    labels = df["pose_label"].values
    groups = df["base_instance_id"].astype(int).values

    return X, labels, scaler, groups


def stratified_group_split(labels: np.ndarray, groups: np.ndarray, test_fraction: float, random_state: int = 42):
    data = pd.DataFrame({'group': groups, 'label': labels})
    group_labels = data.drop_duplicates('group')

    group_counts = group_labels.groupby('label')['group'].nunique()
    insufficient = group_counts[group_counts < 2]
    if not insufficient.empty:
        details = ', '.join(f"{label}: {count}" for label, count in insufficient.items())
        raise ValueError(f"Can them mau cho cac pose -> {details}")

    rng = np.random.default_rng(random_state)
    test_groups = []

    for pose, pose_groups in group_labels.groupby('label')['group']:
        pose_group_ids = pose_groups.to_numpy()
        total_groups = len(pose_group_ids)
        n_test = max(1, int(round(total_groups * test_fraction)))
        if n_test >= total_groups:
            n_test = total_groups - 1
        selected = rng.choice(pose_group_ids, size=n_test, replace=False)
        test_groups.extend(selected.tolist())

    test_mask = np.isin(groups, test_groups)
    test_idx = np.flatnonzero(test_mask)
    train_idx = np.flatnonzero(~test_mask)
    return train_idx, test_idx


# === Grid search utilities ===
def build_fine_values(best_value, multipliers):
    if isinstance(best_value, str):
        return [best_value]
    fine_set = set()
    for m in multipliers:
        candidate = best_value * m
        if candidate > 0:
            fine_set.add(candidate)
    return sorted(fine_set)

def get_adaptive_search_params(pose, total_samples, positive_samples, static_gestures=None):
    """
    Adaptive search strategy based on pose characteristics:
    - Static poses: Simpler search (RBF only)
    - Dynamic poses: More comprehensive search 
    - High imbalance: Focused search with balanced weights
    """
    imbalance_ratio = total_samples / positive_samples if positive_samples > 0 else 1
    
    # Auto-detect static gestures - more flexible than hardcoding
    if static_gestures is None:
        static_gestures = []  # Fallback if not provided
    
    # Static gestures (auto-detected from system) - simpler search
    if pose in static_gestures:
        return {
            'kernels': ['rbf'],
            'Cs': [0.1, 1, 10, 100],
            'gammas': ['auto', 0.01, 0.1, 1],
            'strategy': 'static_optimized'
        }
    
    # High imbalance poses (> 6:1 ratio) - focused search
    elif imbalance_ratio > 6:
        return {
            'kernels': ['rbf'],  # RBF works best for imbalanced
            'Cs': [0.01, 0.1, 1, 10, 100],  # More C values for imbalanced
            'gammas': ['auto', 0.001, 0.01, 0.1, 1],
            'strategy': 'imbalanced_focused'
        }
    
    # Dynamic gestures with good balance - comprehensive search
    else:
        return {
            'kernels': ['rbf', 'linear'],  # 2 best kernels
            'Cs': [0.01, 0.1, 1, 10, 100],
            'gammas': ['auto', 0.001, 0.01, 0.1, 1],
            'strategy': 'balanced_comprehensive'
        }

def run_adaptive_grid_search(pose, estimator, X, y, groups, output_name, static_gestures=None):
    """
    Adaptive grid search that adjusts strategy per pose
    """
    total_samples = len(y)
    positive_samples = y.sum()
    
    params = get_adaptive_search_params(pose, total_samples, positive_samples, static_gestures)
    
    print(f"\n=== Adaptive GridSearch for {pose} ===")
    print(f"Samples: {positive_samples}/{total_samples} (ratio 1:{total_samples/positive_samples:.1f})")
    print(f"Strategy: {params['strategy']}")
    print(f"Search space: {len(params['kernels'])} kernels x {len(params['Cs'])} C x {len(params['gammas'])} gamma = {len(params['kernels']) * len(params['Cs']) * len(params['gammas'])} combinations")
    
    param_grid = {
        "kernel": params['kernels'],
        "C": params['Cs'],
        "gamma": params['gammas'],
    }
    
    cv = GroupKFold(n_splits=min(10, len(np.unique(groups))))  # Adaptive CV folds
    grid = GridSearchCV(
        estimator,
        param_grid,
        cv=cv,
        scoring="f1",  # F1 better for imbalanced data
        n_jobs=2,  # Limit to 2 cores to prevent system freeze
        verbose=1  # Show progress
    )
    
    import time
    start_time = time.time()
    grid.fit(X, y, groups=groups)
    elapsed_time = time.time() - start_time
    
    results = pd.DataFrame(grid.cv_results_).sort_values("mean_test_score", ascending=False)
    display_cols = ["mean_test_score", "std_test_score", "param_kernel", "param_C", "param_gamma"]
    
    print(f"Training time: {elapsed_time:.1f}s")
    print("Top 5 combinations:")
    print(results[display_cols].head(5).to_string(index=False))
    
    results_path = RESULTS_DIR / output_name
    results.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")
    
    return grid, results


def run_grid_search(description: str,
                    estimator: SVC,
                    X: np.ndarray,
                    y: np.ndarray,
                    groups: np.ndarray,
                    kernels,
                    Cs,
                    gammas,
                    output_name: str):
    print(f"\n=== {description} ===")
    param_grid = {
        "kernel": kernels,
        "C": Cs,
        "gamma": gammas,
    }
    cv = GroupKFold(n_splits=10)
    grid = GridSearchCV(
        estimator,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=2,  # Limit to 2 cores to prevent system freeze
    )
    grid.fit(X, y, groups=groups)

    results = pd.DataFrame(grid.cv_results_).sort_values("mean_test_score", ascending=False)
    display_cols = ["mean_test_score", "std_test_score", "param_kernel", "param_C", "param_gamma"]
    print("Top 10 combinations:")
    print(results[display_cols].head(10).to_string(index=False))

    results_path = RESULTS_DIR / output_name
    results.to_csv(results_path, index=False)
    print(f"[INFO] Saved grid results to {results_path}")

    return grid, results


# === Gesture Type Detection ===
def auto_detect_gesture_types(df):
    """Auto-detect STATIC vs DYNAMIC gestures based on delta magnitude"""
    df = df.copy()
    df['delta_magnitude'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2)
    
    static_gestures = []
    dynamic_gestures = []
    
    print("\n=== AUTO-DETECTING GESTURE TYPES ===")
    
    for gesture in sorted(df['pose_label'].unique()):
        gesture_data = df[df['pose_label'] == gesture]
        median_delta = gesture_data['delta_magnitude'].median()
        sample_count = len(gesture_data)
        
        if median_delta < STATIC_THRESHOLD:
            static_gestures.append(gesture)
            gesture_type = "STATIC"
        else:
            dynamic_gestures.append(gesture)
            gesture_type = "DYNAMIC"
        
        print(f"  {gesture:<15} samples={sample_count:<4} median_delta={median_delta:.4f} -> {gesture_type}")
    
    # Apply manual overrides if specified
    if MANUAL_STATIC_OVERRIDE is not None:
        static_gestures = MANUAL_STATIC_OVERRIDE
        print(f"[INFO] Manual override STATIC: {static_gestures}")
    
    if MANUAL_DYNAMIC_OVERRIDE is not None:
        dynamic_gestures = MANUAL_DYNAMIC_OVERRIDE  
        print(f"[INFO] Manual override DYNAMIC: {dynamic_gestures}")
    
    print(f"\nFinal classification:")
    print(f"  STATIC gestures ({len(static_gestures)}): {static_gestures}")
    print(f"  DYNAMIC gestures ({len(dynamic_gestures)}): {dynamic_gestures}")
    
    return static_gestures, dynamic_gestures

# === Main workflows ===
def train_static_dynamic_classifier(X, labels, groups, train_idx, test_idx, scaler, df):
    """Train binary classifier for STATIC vs DYNAMIC gestures"""
    print("\n=== TRAINING STATIC/DYNAMIC CLASSIFIER ===")
    
    # Auto-detect gesture types
    static_gestures, dynamic_gestures = auto_detect_gesture_types(df)
    
    # Assign motion type labels based on auto-detection
    motion_labels = np.array(['STATIC' if label in static_gestures else 'DYNAMIC' for label in labels])
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = motion_labels[train_idx], motion_labels[test_idx]
    
    # Add delta magnitude feature for better classification
    df_temp = pd.DataFrame(X[:, -8:], columns=MOTION_COLS + ["motion_left", "motion_right", "motion_up", "motion_down"])
    delta_magnitude = np.sqrt((df_temp['delta_x'] / DELTA_WEIGHT)**2 + (df_temp['delta_y'] / DELTA_WEIGHT)**2)
    
    # Combine finger states + delta magnitude
    finger_features = X[:, :10]  # First 10 are finger states
    enhanced_features = np.column_stack([finger_features, delta_magnitude])
    
    X_train_enhanced = enhanced_features[train_idx]
    X_test_enhanced = enhanced_features[test_idx]
    
    # Scale the enhanced features
    static_scaler = StandardScaler()
    X_train_scaled = static_scaler.fit_transform(X_train_enhanced)
    X_test_scaled = static_scaler.transform(X_test_enhanced)
    
    # Train SVM
    static_model = SVC(kernel='rbf', probability=True, random_state=42)
    static_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = static_model.score(X_train_scaled, y_train)
    test_acc = static_model.score(X_test_scaled, y_test)
    
    print(f"Static/Dynamic - Train: {train_acc:.3f}, Test: {test_acc:.3f}")
    
    # Save static/dynamic classifier
    static_data = {
        'model': static_model,
        'scaler': static_scaler,
        'feature_cols': [f'left_finger_state_{i}' for i in range(5)] + 
                       [f'right_finger_state_{i}' for i in range(5)] + ['delta_magnitude'],
        'static_gestures': static_gestures,
        'dynamic_gestures': dynamic_gestures,
        'static_threshold': STATIC_THRESHOLD
    }
    
    with open(STATIC_DYNAMIC_PKL, 'wb') as f:
        pickle.dump(static_data, f)
    
    print(f"[INFO] Static/Dynamic classifier saved to {STATIC_DYNAMIC_PKL}")
    return static_model, static_scaler

def train_multiclass(X, labels, groups, train_idx, test_idx, scaler):
    print("=== MULTICLASS TRAINING ===")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    train_groups = groups[train_idx]
    test_groups = groups[test_idx]

    print(f"[INFO] Hold-out test groups: {len(np.unique(test_groups))}")
    print(f"[INFO] CV train groups: {len(np.unique(train_groups))}")

    estimator = SVC(probability=True)
    coarse_grid, coarse_results = run_grid_search(
        "Coarse GridSearch (multiclass)",
        estimator,
        X_train,
        y_train,
        train_groups,
        kernels=["linear", "poly", "rbf", "sigmoid"],
        Cs=COARSE_C_VALUES,
        gammas=COARSE_GAMMA_VALUES,
        output_name="grid_results_coarse_multiclass.csv",
    )

    best_params = coarse_grid.best_params_
    best_kernel = best_params["kernel"]
    best_c = best_params["C"]
    best_gamma = best_params["gamma"]

    print(f"\n[INFO] Best coarse params -> kernel: {best_kernel}, C: {best_c}, gamma: {best_gamma}")

    fine_cs = build_fine_values(best_c, FINE_MULTIPLIERS)
    fine_gammas = build_fine_values(best_gamma, FINE_MULTIPLIERS)

    fine_grid, fine_results = run_grid_search(
        "Fine GridSearch (multiclass)",
        SVC(probability=True),
        X_train,
        y_train,
        train_groups,
        kernels=[best_kernel],
        Cs=fine_cs,
        gammas=fine_gammas,
        output_name="grid_results_fine_multiclass.csv",
    )

    best_model = fine_grid.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    all_label_indices = np.arange(len(label_encoder.classes_))
    report = classification_report(
        y_test,
        y_pred,
        labels=all_label_indices,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
    print("\n=== Evaluation on Hold-out Test (multiclass) ===")
    print(report)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=all_label_indices))

    with open(MODEL_PKL, "wb") as f:
        pickle.dump({
            "model": best_model,
            "label_encoder": label_encoder,
            "finger_cols": LEFT_COLS + RIGHT_COLS,
            "motion_cols": MOTION_COLS,
            "delta_weight": DELTA_WEIGHT,
            "min_delta_mag": MIN_DELTA_MAG,
            "group_column": "base_instance_id",
            "coarse_results": str((RESULTS_DIR / "grid_results_coarse_multiclass.csv").resolve()),
            "fine_results": str((RESULTS_DIR / "grid_results_fine_multiclass.csv").resolve()),
        }, f)

    with open(SCALER_PKL, "wb") as f:
        pickle.dump(scaler, f)

    print("\n[INFO] Multiclass model and scaler have been saved.")

    return label_encoder, y_test, y_pred, best_model


def evaluate_pose_binary(X, labels, groups, train_idx, test_idx, label_encoder, static_gestures=None):
    print("\n=== PER-POSE ONE-VS-REST EVALUATION ===")
    poses = np.unique(labels)

    X_train, X_test = X[train_idx], X[test_idx]
    train_groups = groups[train_idx]
    test_groups = groups[test_idx]

    # Display detected gesture types for transparency
    if static_gestures:
        print(f"Auto-detected STATIC gestures: {static_gestures}")
        dynamic_gestures = [pose for pose in poses if pose not in static_gestures]
        print(f"Auto-detected DYNAMIC gestures: {dynamic_gestures}")
    
    summary_rows = []

    for pose in poses:
        print(f"\n--- Pose: {pose} ---")
        y_binary = (labels == pose).astype(int)
        y_train = y_binary[train_idx]
        y_test = y_binary[test_idx]

        positives = y_train.sum()
        negatives = len(y_train) - positives
        if positives == 0 or negatives == 0:
            print("[WARN] Not enough data for binary classification. Skipping.")
            continue

        estimator = SVC(class_weight='balanced', probability=True, random_state=42)
        
        # Use adaptive grid search strategy per pose
        grid, _ = run_adaptive_grid_search(
            pose,
            estimator,
            X_train,
            y_train,
            train_groups,
            output_name=f"adaptive_grid_results_{pose}.csv",
            static_gestures=static_gestures
        )

        best_params = grid.best_params_
        best_kernel = best_params["kernel"]
        best_c = best_params["C"]
        best_gamma = best_params["gamma"]
        best_score = grid.best_score_
        
        print(f"OPTIMAL PARAMS for {pose}:")
        print(f"   Kernel: {best_kernel}")
        print(f"   C: {best_c}")
        print(f"   Gamma: {best_gamma}")
        print(f"   CV F1-Score: {best_score:.4f}")
        
        # Train final model with best params
        fine_grid = grid  # Use the already trained best model

        best_model = fine_grid.best_estimator_
        # Model already trained by GridSearchCV
        y_pred = best_model.predict(X_test)

        labels_order = [0, 1]
        target_names = ['other', pose]
        report_dict = classification_report(
            y_test,
            y_pred,
            labels=labels_order,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        report_text = classification_report(
            y_test,
            y_pred,
            labels=labels_order,
            target_names=target_names,
            zero_division=0,
        )
        print(report_text)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=labels_order))

        # Determine gesture type for this pose
        gesture_type = 'STATIC' if pose in (static_gestures or []) else 'DYNAMIC'
        
        # Get search strategy used for this pose
        pose_params = get_adaptive_search_params(pose, len(y_test), y_test.sum(), static_gestures)
        search_strategy = pose_params.get('strategy', 'unknown')
        
        pose_metrics = {
            'pose_label': pose,
            'gesture_type': gesture_type,
            'test_samples': int(len(y_test)),
            'positive_samples': int(y_test.sum()),
            'imbalance_ratio': f"1:{len(y_test)/y_test.sum():.1f}",
            'search_strategy': search_strategy,
            'best_kernel': best_kernel,
            'best_C': best_c,
            'best_gamma': best_gamma,
            'cv_f1_score': best_score,
            'test_accuracy': report_dict['accuracy'],
            'test_precision': report_dict[pose]['precision'],
            'test_recall': report_dict[pose]['recall'],
            'test_f1_score': report_dict[pose]['f1-score'],
        }
        summary_rows.append(pose_metrics)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        
        # Display comprehensive results
        print("\n" + "="*80)
        print("OPTIMAL HYPERPARAMETERS SUMMARY FOR EACH POSE_LABEL")
        print("="*80)
        
        # Group by strategy for better visualization
        for pose_idx, row in summary_df.iterrows():
            pose = row['pose_label']
            gesture_type = row['gesture_type']
            strategy = row['search_strategy']
            print(f"\n{pose.upper()} ({gesture_type}):")
            print(f"   Data: {row['positive_samples']}/{row['test_samples']} samples ({row['imbalance_ratio']})")
            print(f"   Strategy: {strategy}")
            print(f"   Optimal: kernel={row['best_kernel']}, C={row['best_C']}, gamma={row['best_gamma']}")
            print(f"   Performance: CV_F1={row['cv_f1_score']:.3f}, Test_F1={row['test_f1_score']:.3f}")
            print(f"   Precision={row['test_precision']:.3f}, Recall={row['test_recall']:.3f}")
        
        # Save detailed results
        summary_path = RESULTS_DIR / "optimal_hyperparameters_per_pose.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nDetailed hyperparameters saved to: {summary_path}")
        
        # Create hyperparameters-only table for easy reference
        hyperparams_df = summary_df[['pose_label', 'best_kernel', 'best_C', 'best_gamma', 'cv_f1_score']].copy()
        hyperparams_path = RESULTS_DIR / "best_hyperparameters_lookup.csv"
        hyperparams_df.to_csv(hyperparams_path, index=False)
        print(f"Quick lookup table saved to: {hyperparams_path}")
        
        print("\nHYPERPARAMETER PATTERNS:")
        kernel_counts = summary_df['best_kernel'].value_counts()
        for kernel, count in kernel_counts.items():
            print(f"   {kernel}: {count}/{len(summary_df)} poses ({count/len(summary_df)*100:.1f}%)")
            
        print(f"\nAverage CV F1-Score: {summary_df['cv_f1_score'].mean():.3f}")
        print(f"Average Test F1-Score: {summary_df['test_f1_score'].mean():.3f}")
        print("="*80)


def report_full_dataset(model, label_encoder, X_full, labels_full):
    print("\n=== FULL DATASET EVALUATION ===")
    y_true = label_encoder.transform(labels_full)
    y_pred = model.predict(X_full)
    target_names = label_encoder.classes_
    print(classification_report(y_true, y_pred, target_names=target_names))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("\nPer-pose accuracy:")
    for idx, pose in enumerate(target_names):
        total = cm[idx].sum()
        correct = cm[idx, idx]
        if total == 0:
            accuracy = 0.0
        else:
            accuracy = correct / total
        print(f"  {pose:15s} total={total:4d} correct={correct:4d} wrong={total - correct:3d} accuracy={accuracy*100:5.1f}%")


# === Main ===
def main(dataset_path: str = DEFAULT_DATASET):
    print("=== TRAIN MOTION SVM WITH FINGER CONTEXT ===")
    print(f"[INFO] Using dataset: {dataset_path}")

    # Auto-detect dataset - try new data first, fallback to old
    datasets_to_try = [
        "gesture_data_09_10_2025.csv",  # New real data
        dataset_path,                   # Fallback to default
    ]
    
    df = None
    for dataset in datasets_to_try:
        try:
            df = load_dataset(dataset)
            print(f"[INFO] Successfully loaded: {dataset}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        raise FileNotFoundError("No valid dataset found!")

    X, labels, scaler, groups = prepare_features(df)
    train_idx, test_idx = stratified_group_split(labels, groups, test_fraction=TEST_FRACTION, random_state=42)

    # Train Static/Dynamic classifier first and get detected gesture types
    static_model, static_scaler = train_static_dynamic_classifier(X, labels, groups, train_idx, test_idx, scaler, df)
    
    # Auto-detect static gestures for adaptive strategy
    static_gestures, dynamic_gestures = auto_detect_gesture_types(df)
    
    # Then train main multiclass model
    label_encoder, y_test_enc, y_pred_enc, best_model = train_multiclass(X, labels, groups, train_idx, test_idx, scaler)
    evaluate_pose_binary(X, labels, groups, train_idx, test_idx, label_encoder, static_gestures)
    report_full_dataset(best_model, label_encoder, X, labels)
    
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Static/Dynamic classifier: {STATIC_DYNAMIC_PKL}")
    print(f"Main gesture classifier: {MODEL_PKL}")
    print(f"Feature scaler: {SCALER_PKL}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train motion SVM models with finger context.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to the merged dataset CSV.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(dataset_path=args.dataset)
