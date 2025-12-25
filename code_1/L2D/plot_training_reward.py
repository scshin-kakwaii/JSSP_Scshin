import matplotlib.pyplot as plt
import ast
import os
import numpy as np

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================

# 1. Tên file log training (Hãy sửa tên này trùng với file bạn đang có)
# Ví dụ: log_6_6_1_99.txt
INPUT_LOG_FILE = 'code_1/L2D/log_6_6_1_99.txt'

# 2. Đường dẫn tuyệt đối để lưu ảnh (Như bạn yêu cầu)
SAVE_PATH = '/Users/shenchiashin/Downloads/CAPSTONE_3RD MODIFICATION/code_1/L2D/chart_training_reward.png'

# 3. Độ làm mượt (Số càng lớn đường càng phẳng)
SMOOTH_WINDOW = 50  

# ==========================================

def plot_training_reward(filename, save_path):
    # Kiểm tra file input
    if not os.path.exists(filename):
        # Thử tìm trong cùng thư mục với SAVE_PATH xem sao
        alternative_path = os.path.join(os.path.dirname(save_path), filename)
        if os.path.exists(alternative_path):
            filename = alternative_path
        else:
            print(f"❌ LỖI: Không tìm thấy file log '{filename}'")
            print(f"👉 Hãy copy file log vào cùng thư mục chạy code hoặc sửa lại đường dẫn INPUT_LOG_FILE.")
            return

    try:
        print(f"📖 Đang đọc file: {filename}...")
        with open(filename, 'r') as f:
            content = f.read()
            # Chuyển chuỗi string thành list
            data = ast.literal_eval(content)
            
        # Tách dữ liệu
        episodes = np.array([x[0] for x in data])
        rewards = np.array([x[1] for x in data])
        
        print(f"📊 Đã đọc được {len(episodes)} điểm dữ liệu.")

        # --- VẼ BIỂU ĐỒ ---
        plt.figure(figsize=(12, 6))
        
        # 1. Vẽ Reward gốc (Mờ) - Để thấy độ dao động thực tế
        plt.plot(episodes, rewards, color='gray', alpha=0.3, linewidth=0.5, label='Raw Reward')
        
        # 2. Vẽ Trend (Làm mượt) - Để thấy xu hướng học
        if len(rewards) >= SMOOTH_WINDOW:
            window = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
            smooth_rewards = np.convolve(rewards, window, mode='valid')
            # Cắt bớt trục X cho khớp độ dài sau khi làm mượt
            smooth_eps = episodes[SMOOTH_WINDOW-1:]
            
            plt.plot(smooth_eps, smooth_rewards, color='#1f77b4', linewidth=2, label=f'Trend (MA-{SMOOTH_WINDOW})')

        # Trang trí
        plt.title('Training Progress (Reward Evolution)', fontsize=16, fontweight='bold')
        plt.xlabel('Training Updates', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')
        
        plt.tight_layout()
        
        # --- LƯU FILE ---
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300)
        print(f"✅ THÀNH CÔNG! Biểu đồ đã được lưu tại:")
        print(f"👉 {save_path}")
        
        # Tự động mở ảnh trên Mac
        os.system(f"open '{save_path}'")

    except Exception as e:
        print(f"❌ Lỗi khi xử lý: {e}")

if __name__ == '__main__':
    plot_training_reward(INPUT_LOG_FILE, SAVE_PATH)
