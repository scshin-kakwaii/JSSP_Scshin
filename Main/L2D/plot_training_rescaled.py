import matplotlib.pyplot as plt
import ast
import os
import numpy as np

# --- CẤU HÌNH ---
# Sửa tên file log của bạn vào đây
LOG_FILE = 'code_1/L2D/log_6_6_1_99.txt' 

# Độ làm mượt (Trend)
WINDOW_SIZE = 50

def plot_rescaled(filename):
    if not os.path.exists(filename):
        print(f"❌ Không tìm thấy file {filename}")
        return

    try:
        with open(filename, 'r') as f:
            content = f.read()
            data = ast.literal_eval(content)
            
        episodes = np.array([x[0] for x in data])
        
        # --- THAY ĐỔI Ở ĐÂY: NHÂN 100 ĐỂ VỀ LẠI SCALE GỐC ---
        # Reward lúc train bị chia 100, giờ nhân lại để dễ nhìn
        rewards = np.array([x[1] * 100 for x in data])
        # ----------------------------------------------------
        
        plt.figure(figsize=(10, 6))
        
        # 1. Vẽ dữ liệu gốc (Mờ)
        plt.plot(episodes, rewards, color='silver', alpha=0.4, linewidth=0.5, label='Raw Reward (Rescaled)')
        
        # 2. Vẽ đường xu hướng (Đậm)
        if len(rewards) >= WINDOW_SIZE:
            window = np.ones(WINDOW_SIZE) / WINDOW_SIZE
            smooth_rewards = np.convolve(rewards, window, mode='valid')
            smooth_eps = episodes[WINDOW_SIZE-1:]
            
            # Dùng màu đỏ hoặc xanh đậm để nổi bật
            plt.plot(smooth_eps, smooth_rewards, color='#0052cc', linewidth=2.5, label=f'Trend (Average Score)')

        # Trang trí
        plt.title(f'Training Progress (Rescaled x100)', fontsize=16, fontweight='bold')
        plt.xlabel('Training Updates', fontsize=14)
        
        # Đổi tên trục Y cho hợp lý
        plt.ylabel('Estimated Score (~Negative Step Cost)', fontsize=14)
        
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend(fontsize=12, loc='lower right', frameon=True)
        
        plt.tight_layout()
        plt.savefig('chart_training_rescaled.png', dpi=300)
        print("✅ Đã lưu biểu đồ (đã nhân 100): chart_training_rescaled.png")
        plt.show()

    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == '__main__':
    plot_rescaled(LOG_FILE)