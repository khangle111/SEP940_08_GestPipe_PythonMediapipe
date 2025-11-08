#!/usr/bin/env python3
"""
User-specific training script for gesture pipeline
Based on train_motion_svm_all_models.py but optimized for user datasets
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

def create_balanced_from_compact(compact_file, output_file, samples_per_gesture=100):
    """Create balanced dataset from compact file for training"""
    print(f"Loading compact data from: {compact_file}")
    
    # Load compact data
    df = pd.read_csv(compact_file)
    
    print(f"Original data: {len(df)} samples")
    print(f"Gestures: {df['pose_label'].unique().tolist()}")
    
    # Create balanced dataset
    balanced_data = []
    instance_id = 1
    
    for gesture in df['pose_label'].unique():
        print(f"Processing {gesture}...")
        
        # Get sample for this gesture
        gesture_sample = df[df['pose_label'] == gesture].iloc[0]
        
        # Create 100 variations with small noise
        for i in range(samples_per_gesture):
            new_row = gesture_sample.copy()
            new_row['instance_id'] = instance_id
            
            # Add small realistic noise to motion coordinates only
            motion_cols = ['motion_x_start', 'motion_y_start', 'motion_x_mid', 
                           'motion_y_mid', 'motion_x_end', 'motion_y_end']
            
            for col in motion_cols:
                if col in new_row:
                    noise = np.random.normal(0, 0.015)
                    new_row[col] = max(0.0, min(1.0, new_row[col] + noise))
            
            # Keep finger states exactly the same (most important!)
            balanced_data.append(new_row)
            instance_id += 1
        
        print(f"  Generated {samples_per_gesture} samples for {gesture}")
    
    # Convert to DataFrame and save
    balanced_df = pd.DataFrame(balanced_data)
    balanced_df.to_csv(output_file, index=False)
    print(f"Balanced dataset saved to: {output_file}")
    
    return balanced_df

def load_and_prepare_data(dataset_path):
    """Load and prepare data for training"""
    print(f"Loading dataset: {dataset_path}")
    
    # Check if it's compact data that needs balancing
    df = pd.read_csv(dataset_path)
    
    # If dataset is small (likely compact), create balanced version
    if len(df) <= 20:  # Compact dataset
        print("Detected compact dataset, creating balanced version...")
        # Save balanced dataset in user's training_results folder
        dataset_path_obj = Path(dataset_path)
        if len(dataset_path_obj.parts) > 1 and dataset_path_obj.parts[0].startswith('user_'):
            user_folder = Path(dataset_path_obj.parts[0])  # Just the user folder name
            balanced_path = user_folder / 'training_results' / 'balanced_dataset.csv'
        else:
            balanced_path = str(dataset_path).replace('.csv', '_balanced.csv')
        df = create_balanced_from_compact(dataset_path, balanced_path)
    
    print(f"Total samples: {len(df)}")
    
    # Create motion direction features matching train_motion_svm_all_models.py 
    DELTA_WEIGHT = 15.0
    df['motion_left'] = (df['delta_x'] < 0).astype(float) * DELTA_WEIGHT
    df['motion_right'] = (df['delta_x'] > 0).astype(float) * DELTA_WEIGHT
    df['motion_up'] = (df['delta_y'] < 0).astype(float) * DELTA_WEIGHT
    df['motion_down'] = (df['delta_y'] > 0).astype(float) * DELTA_WEIGHT
    
    # Apply delta weight to motion columns
    df['delta_x'] = df['delta_x'] * DELTA_WEIGHT
    df['delta_y'] = df['delta_y'] * DELTA_WEIGHT
    
    # Prepare features and labels - matching train_motion_svm_all_models.py exactly (18 features)
    feature_columns = [
        'left_finger_state_0', 'left_finger_state_1', 'left_finger_state_2', 'left_finger_state_3', 'left_finger_state_4',
        'right_finger_state_0', 'right_finger_state_1', 'right_finger_state_2', 'right_finger_state_3', 'right_finger_state_4',
        'main_axis_x', 'main_axis_y', 'delta_x', 'delta_y', 'motion_left', 'motion_right', 'motion_up', 'motion_down'
    ]
    
    X = df[feature_columns].values
    y = df['pose_label'].values
    
    print(f"Features shape: {X.shape}")
    print(f"Labels: {np.unique(y)}")
    
    return X, y, df

def train_static_dynamic_classifier(X, y):
    """Train static/dynamic classifier"""
    print("\n=== TRAINING STATIC/DYNAMIC CLASSIFIER ===")
    
    # Define static gestures (minimal motion)
    static_gestures = ['home', 'end']
    
    # Create binary labels for static/dynamic
    y_static = np.array(['static' if label in static_gestures else 'dynamic' for label in y])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_static, test_size=0.2, random_state=42, stratify=y_static)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    svm = SVC(kernel='rbf', C=10, gamma='auto', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = svm.score(X_train_scaled, y_train)
    test_score = svm.score(X_test_scaled, y_test)
    print(f"Static/Dynamic - Train: {train_score:.3f}, Test: {test_score:.3f}")
    
    return svm, scaler

def train_gesture_classifier(X, y):
    """Train main gesture classifier"""
    print("\n=== TRAINING GESTURE CLASSIFIER ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Grid search for best parameters
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf']
    }
    
    print("Running GridSearchCV...")
    grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='f1_macro')
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.3f}")
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test_scaled)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Test F1-score: {test_f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return best_model, scaler

def main():
    parser = argparse.ArgumentParser(description='Train user-specific gesture models')
    parser.add_argument('--dataset', required=True, help='Path to user dataset CSV file')
    args = parser.parse_args()
    
    # Determine user folder from dataset path
    dataset_path = Path(args.dataset)
    if len(dataset_path.parts) > 1 and dataset_path.parts[0].startswith('user_'):
        # User-specific path (e.g., user_Khang/training_results/gesture_data_compact.csv)
        user_folder = Path(dataset_path.parts[0])  # Extract user_Khang from path
        models_dir = user_folder / 'models'
        training_results_dir = user_folder / 'training_results'
    else:
        # Default path
        models_dir = Path('models')
        training_results_dir = Path('training_results')
    
    # Create directories
    models_dir.mkdir(exist_ok=True)
    training_results_dir.mkdir(exist_ok=True)
    
    # Load data
    X, y, df = load_and_prepare_data(args.dataset)
    
    # Train static/dynamic classifier
    static_dynamic_model, static_dynamic_scaler = train_static_dynamic_classifier(X, y)
    
    # Train main gesture classifier
    gesture_model, gesture_scaler = train_gesture_classifier(X, y)
    
    # Save models
    static_dynamic_path = models_dir / 'static_dynamic_classifier.pkl'
    gesture_model_path = models_dir / 'motion_svm_model.pkl'
    gesture_scaler_path = models_dir / 'motion_scaler.pkl'
    
    joblib.dump(static_dynamic_model, static_dynamic_path)
    joblib.dump(gesture_model, gesture_model_path)
    joblib.dump(gesture_scaler, gesture_scaler_path)
    
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Static/Dynamic classifier: {static_dynamic_path}")
    print(f"Main gesture classifier: {gesture_model_path}")
    print(f"Feature scaler: {gesture_scaler_path}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)