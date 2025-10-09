#!/usr/bin/env python3
"""
Script để xem độ chính xác từng pose_label từ kết quả training đã có
"""
import os
import pickle
import pandas as pd
import numpy as np

def load_and_analyze_results():
    """Tải và phân tích kết quả training"""
    
    print("🔍 PHÂN TÍCH ĐỘ CHÍNH XÁC TỪNG POSE_LABEL")
    print("=" * 60)
    
    # 1. Xem kết quả từ pose_binary_summary.csv
    summary_file = "pose_binary_summary.csv"
    if os.path.exists(summary_file):
        print("\n📊 KẾT QUẢ BINARY CLASSIFICATION (One-vs-Rest):")
        print("-" * 50)
        df_summary = pd.read_csv(summary_file)
        
        # Format và hiển thị đẹp
        print(f"{'Pose Label':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Samples':<8}")
        print("-" * 60)
        
        for _, row in df_summary.iterrows():
            pose = row['pose_label']
            precision = f"{row['precision_pose']*100:.1f}%"
            recall = f"{row['recall_pose']*100:.1f}%"
            f1 = f"{row['f1_pose']*100:.1f}%"
            samples = int(row['positive_samples'])
            
            print(f"{pose:<15} {precision:<10} {recall:<10} {f1:<10} {samples:<8}")
    
    # 2. Tải model và kiểm tra multiclass accuracy
    model_file = "motion_svm_model.pkl"
    if os.path.exists(model_file):
        print(f"\n📈 MULTICLASS MODEL INFO:")
        print("-" * 30)
        
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        label_encoder = model_data['label_encoder']
        gestures = label_encoder.classes_
        
        print(f"✅ Tổng số gestures: {len(gestures)}")
        print(f"✅ Danh sách gestures: {list(gestures)}")
        
    # 3. Kiểm tra dataset gốc để có context
    dataset_file = "gesture_data_09_10_2025.csv"
    if os.path.exists(dataset_file):
        print(f"\n📋 DATASET INFO:")
        print("-" * 20)
        
        df = pd.read_csv(dataset_file)
        pose_counts = df['pose_label'].value_counts().sort_index()
        
        print(f"✅ Tổng samples: {len(df)}")
        print(f"✅ Distribution per gesture:")
        
        for pose, count in pose_counts.items():
            print(f"   {pose:<15}: {count:>4} samples")
    
    # 4. Xem kết quả grid search tốt nhất
    grid_file = "grid_results_fine_multiclass.csv"
    if os.path.exists(grid_file):
        print(f"\n🎯 BEST MULTICLASS PARAMETERS:")
        print("-" * 35)
        
        grid_df = pd.read_csv(grid_file)
        best_result = grid_df.iloc[0]  # Đã sorted theo mean_test_score
        
        print(f"✅ Best CV Accuracy: {best_result['mean_test_score']*100:.2f}%")
        print(f"✅ Std deviation: ±{best_result['std_test_score']*100:.2f}%")
        print(f"✅ Best kernel: {best_result['param_kernel']}")
        print(f"✅ Best C: {best_result['param_C']}")
        print(f"✅ Best gamma: {best_result['param_gamma']}")

def main():
    """Hàm chính"""
    try:
        load_and_analyze_results()
        print(f"\n🎉 PHÂN TÍCH HOÀN TẤT!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("💡 Hãy chờ training hoàn thành hoặc kiểm tra file kết quả")

if __name__ == "__main__":
    main()