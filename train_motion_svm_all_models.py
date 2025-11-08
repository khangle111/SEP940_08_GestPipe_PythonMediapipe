import argparse
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_and_prepare_data(csv_path):
    """
    Load CSV data and prepare features for training
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        X: Feature matrix
        y: Labels
        feature_names: List of feature column names
    """
    print(f"Loading data from: {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} samples")
    print(f"   Gestures: {list(df['pose_label'].unique())}")
    
    # Define feature columns
    finger_cols = [
        'left_finger_state_0', 'left_finger_state_1', 'left_finger_state_2', 
        'left_finger_state_3', 'left_finger_state_4',
        'right_finger_state_0', 'right_finger_state_1', 'right_finger_state_2', 
        'right_finger_state_3', 'right_finger_state_4'
    ]
    
    motion_cols = [
        'motion_x_start', 'motion_y_start', 'motion_x_mid', 
        'motion_y_mid', 'motion_x_end', 'motion_y_end'
    ]
    
    feature_cols = ['main_axis_x', 'main_axis_y', 'delta_x', 'delta_y']
    
    # Combine all features
    all_features = finger_cols + motion_cols + feature_cols
    
    # Check if all columns exist
    missing_cols = [col for col in all_features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")
    
    # Prepare features and labels
    X = df[all_features].values
    y = df['pose_label'].values
    
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Number of classes: {len(np.unique(y))}")
    
    return X, y, all_features


def train_svm_model(X, y, output_path, model_name="gesture_svm"):
    """
    Train SVM model with hyperparameter tuning
    
    Args:
        X: Feature matrix
        y: Labels
        output_path: Output directory path
        model_name: Base name for model files
    
    Returns:
        model: Trained SVM model
        scaler: Fitted StandardScaler
        accuracy: Model accuracy on test set
    """
    print(f"Training SVM model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly']
    }
    
    print(f"   Performing hyperparameter tuning...")
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(
        svm, param_grid, cv=5, scoring='accuracy', 
        n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    print(f"   Best parameters: {grid_search.best_params_}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model training completed!")
    print(f"   Test accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and scaler
    model_path = Path(output_path) / f"{model_name}.pkl"
    scaler_path = Path(output_path) / f"{model_name}_scaler.pkl"
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Model saved: {model_path}")
    print(f"Scaler saved: {scaler_path}")
    
    return best_model, scaler, accuracy


def create_model_info(output_path, model_name, feature_names, accuracy, gestures):
    """Create model information file"""
    
    info_path = Path(output_path) / f"{model_name}_info.txt"
    
    with open(info_path, 'w') as f:
        f.write(f"=== GESTURE RECOGNITION MODEL INFO ===\n\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Training Timestamp: {pd.Timestamp.now()}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        
        f.write(f"Supported Gestures ({len(gestures)}):\n")
        for i, gesture in enumerate(sorted(gestures)):
            f.write(f"  {i+1}. {gesture}\n")
        
        f.write(f"\nFeatures ({len(feature_names)}):\n")
        for i, feature in enumerate(feature_names):
            f.write(f"  {i+1}. {feature}\n")
        
        f.write(f"\nUsage:\n")
        f.write(f"  import joblib\n")
        f.write(f"  model = joblib.load('{model_name}.pkl')\n")
        f.write(f"  scaler = joblib.load('{model_name}_scaler.pkl')\n")
        f.write(f"  \n")
        f.write(f"  # Predict\n")
        f.write(f"  X_scaled = scaler.transform(features)\n")
        f.write(f"  prediction = model.predict(X_scaled)\n")
    
    print(f"Model info saved: {info_path}")


def main():
    parser = argparse.ArgumentParser(description="Train gesture recognition SVM model")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output-dir", default="models", help="Output directory for models")
    parser.add_argument("--model-name", default="gesture_svm", help="Base name for model files")
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data
        X, y, feature_names = load_and_prepare_data(args.input)
        
        # Train model
        model, scaler, accuracy = train_svm_model(
            X, y, output_dir, args.model_name
        )
        
        # Create model info
        gestures = np.unique(y)
        create_model_info(
            output_dir, args.model_name, feature_names, accuracy, gestures
        )
        
        print(f"\nTraining pipeline completed successfully!")
        print(f"Models saved in: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)