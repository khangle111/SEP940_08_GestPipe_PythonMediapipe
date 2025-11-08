#!/usr/bin/env python3
"""
Interactive Gesture Training Script with Arrow Key Navigation
"""

import os
import msvcrt
import sys
import subprocess
from pathlib import Path
import glob

def arrow_menu(title, options, descriptions=None):
    """
    Interactive menu với arrow keys và Enter
    """
    if not options:
        return None
        
    current_index = 0
    
    def print_menu():
        os.system('cls' if os.name == 'nt' else 'clear')
        print(title)
        print("=" * len(title))
        print()
        
        for i, option in enumerate(options):
            prefix = "► " if i == current_index else "  "
            desc = f" - {descriptions[i]}" if descriptions and i < len(descriptions) else ""
            
            if i == current_index:
                print(f"\033[92m{prefix}{option}{desc}\033[0m")  # Green highlight
            else:
                print(f"{prefix}{option}{desc}")
        
        print()
        print("🔼🔽 Mũi tên lên/xuống | ⏎ Enter để chọn | ESC để thoát")
    
    while True:
        print_menu()
        
        # Get key input
        key = msvcrt.getch()
        
        if key == b'\xe0':  # Arrow key prefix
            key = msvcrt.getch()
            if key == b'H':  # Up arrow
                current_index = (current_index - 1) % len(options)
            elif key == b'P':  # Down arrow
                current_index = (current_index + 1) % len(options)
        elif key == b'\r':  # Enter
            return current_index
        elif key == b'\x1b':  # ESC
            return None
        elif key == b'q' or key == b'Q':  # Q to quit
            return None

def yes_no_menu(question):
    """
    Yes/No menu với arrow keys
    """
    options = ["Có (Yes)", "Không (No)"]
    result = arrow_menu(question, options)
    
    if result is None:
        return None
    return result == 0  # True for Yes, False for No

def get_user_folders():
    """Get all user folders with their status"""
    user_info = []
    
    for folder_name in os.listdir('.'):
        if os.path.isdir(folder_name) and folder_name.startswith('user_'):
            username = folder_name[5:]  # Remove 'user_' prefix
            
            # Check for raw data
            raw_data_path = os.path.join(folder_name, 'raw_data')
            models_path = os.path.join(folder_name, 'models')
            
            csv_files = []
            if os.path.exists(raw_data_path):
                csv_files = [f for f in os.listdir(raw_data_path) if f.endswith('.csv')]
            
            has_models = os.path.exists(models_path) and len(os.listdir(models_path)) > 0
            
            status = []
            if csv_files:
                gestures = set()
                for csv_file in csv_files:
                    # Extract gesture name from filename
                    parts = csv_file.split('_')
                    if len(parts) >= 4:
                        gesture = parts[3]
                        gestures.add(gesture)
                status.append(f"{len(csv_files)} CSV files")
                if gestures:
                    status.append(f"Gestures: {', '.join(sorted(gestures))}")
            
            if has_models:
                status.append("Đã có models")
            else:
                status.append("Chưa có models")
            
            user_info.append({
                'username': username,
                'folder': folder_name,
                'csv_count': len(csv_files),
                'has_models': has_models,
                'gestures': list(gestures) if csv_files else [],
                'status': ' | '.join(status)
            })
    
    return user_info

def train_user_models(username):
    """Train models for selected user"""
    print(f"\n🤖 TRAINING MODELS CHO USER: {username}")
    print("="*50)
    
    try:
        # Run user_gesture_pipeline.py
        cmd = [sys.executable, 'user_gesture_pipeline.py', '--user-folder', username]
        print(f"📝 Đang chạy: {' '.join(cmd)}")
        print("⏳ Vui lòng đợi...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print(f"\n✅ THÀNH CÔNG! Models đã được tạo cho {username}")
            
            # Show summary from output
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['success', 'completed', 'model', 'training', 'compact']):
                    print(f"   {line.strip()}")
                    
            return True
        else:
            print(f"\n❌ LỖI TRAINING!")
            print("Error details:")
            error_lines = result.stderr.split('\n')[:10]  # Show first 10 lines
            for line in error_lines:
                if line.strip():
                    print(f"   {line.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n⏰ TIMEOUT! Training quá lâu (>5 phút)")
        return False
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        return False

def main():
    """Main function"""
    try:
        while True:
            # Get user folders
            user_info = get_user_folders()
            
            if not user_info:
                print("📂 Không tìm thấy user folder nào!")
                print("Vui lòng chạy collect_data_update.py để tạo data trước.")
                return
            
            # Prepare options
            options = []
            descriptions = []
            
            for info in user_info:
                options.append(info['username'])
                
                # Create description
                desc_parts = []
                if info['csv_count'] > 0:
                    desc_parts.append(f"{info['csv_count']} CSV")
                    if info['gestures']:
                        desc_parts.append(f"Gestures: {', '.join(info['gestures'][:3])}")
                        if len(info['gestures']) > 3:
                            desc_parts.append("...")
                
                if info['has_models']:
                    desc_parts.append("✅ Có models")
                else:
                    if info['csv_count'] > 0:
                        desc_parts.append("🔄 Cần train")
                    else:
                        desc_parts.append("❌ Không có data")
                
                descriptions.append(" | ".join(desc_parts))
            
            # Add exit option
            options.append("Thoát")
            descriptions.append("Exit chương trình")
            
            # Show menu
            choice = arrow_menu("🤖 GESTURE MODEL TRAINING", options, descriptions)
            
            if choice is None or choice == len(user_info):
                print("\n👋 Thoát chương trình")
                break
            
            selected_user = user_info[choice]
            username = selected_user['username']
            
            # Check if user has data
            if selected_user['csv_count'] == 0:
                print(f"\n❌ User '{username}' không có data để train!")
                input("Nhấn Enter để tiếp tục...")
                continue
            
            # Show details and confirm
            print(f"\n📊 THÔNG TIN USER: {username}")
            print(f"   📁 Folder: {selected_user['folder']}")
            print(f"   📄 CSV files: {selected_user['csv_count']}")
            print(f"   🎯 Gestures: {', '.join(selected_user['gestures'])}")
            print(f"   🤖 Models: {'Có' if selected_user['has_models'] else 'Chưa có'}")
            
            if selected_user['has_models']:
                overwrite = yes_no_menu(f"⚠️  User '{username}' đã có models\n❓ Muốn train lại (overwrite)?")
                if overwrite is None:
                    continue
                elif not overwrite:
                    print("⏸️  Bỏ qua training")
                    input("Nhấn Enter để tiếp tục...")
                    continue
            
            # Confirm training
            confirm = yes_no_menu(f"🚀 Bắt đầu training cho user '{username}'?")
            if confirm is None:
                continue
            elif not confirm:
                continue
            
            # Start training
            success = train_user_models(username)
            
            if success:
                print(f"\n🎉 HOÀN THÀNH! User '{username}' đã được train xong!")
                
                # Check results
                models_path = os.path.join(f"user_{username}", "models")
                if os.path.exists(models_path):
                    model_files = [f for f in os.listdir(models_path) if f.endswith('.pkl')]
                    print(f"📁 Tạo được {len(model_files)} model files:")
                    for model_file in model_files:
                        print(f"   • {model_file}")
                        
                compact_path = os.path.join(f"user_{username}", "gesture_data_compact.csv")
                if os.path.exists(compact_path):
                    print(f"📊 Tạo compact dataset: gesture_data_compact.csv")
            else:
                print(f"\n💥 THẤT BẠI! Có lỗi khi training user '{username}'")
            
            input("\nNhấn Enter để tiếp tục...")
            
    except KeyboardInterrupt:
        print("\n\n👋 Chương trình bị dừng bởi người dùng")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()