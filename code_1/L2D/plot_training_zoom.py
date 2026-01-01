import matplotlib.pyplot as plt
import ast
import os
import numpy as np

# --- CẤU HÌNH ---
LOG_FILE = 'code_1/L2D/log_6_6_1_99.txt' 

# Tăng độ làm mượt lên 500 (để đường trend siêu mịn)
WINDOW_SIZE = 500  

def plot_zoom_trend(filename):
    if not os.path.exists(filename):
        print(f"❌ Không tìm thấy file {filename}")
        return

    try:
        with open(filename, 'r') as f:
            content = f.read()
            data = ast.literal_eval(content)
            
        episodes = np.array([x[0] for x in data])
        # Nhân 100 để về scale thực tế
        rewards = np.array([x[1] * 100 for x in data])
        
        plt.figure(figsize=(12, 7))
        
        # 1. Vẽ dữ liệu gốc (Rất mờ - chỉ để làm nền)
        plt.plot(episodes, rewards, color='lightgray', alpha=0.3, linewidth=0.5, label='Raw Noise')
        
        # 2. Tính toán Trend siêu mượt
        if len(rewards) >= WINDOW_SIZE:
            window = np.ones(WINDOW_SIZE) / WINDOW_SIZE
            smooth_rewards = np.convolve(rewards, window, mode='valid')
            smooth_eps = episodes[WINDOW_SIZE-1:]
            
            # Vẽ đường Trend
            plt.plot(smooth_eps, smooth_rewards, color='#0052cc', linewidth=3, label=f'Macro Trend (MA-{WINDOW_SIZE})')

            # --- KỸ THUẬT ZOOM CẬN CẢNH ---
            # Tìm giá trị thấp nhất và cao nhất CỦA ĐƯỜNG TREND (không phải của raw data)
            min_trend = np.min(smooth_rewards)
            max_trend = np.max(smooth_rewards)
            
            # Thiết lập giới hạn trục Y bám sát vào đường Trend
            # Thêm khoảng đệm 10% để nhìn cho thoáng
            margin = (max_trend - min_trend) * 0.2
            plt.ylim(min_trend - margin, max_trend + margin)
            
            # --- ĐÁNH DẤU SỰ CẢI THIỆN ---
            start_val = smooth_rewards[0]
            end_val = smooth_rewards[-1]
            improvement = end_val - start_val
            
            # Ghi chú lên hình
            plt.annotate(f'Start: {start_val:.1f}', xy=(smooth_eps[0], start_val), 
                         xytext=(smooth_eps[0], start_val - margin/2),
                         arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='red')
                         
            plt.annotate(f'End: {end_val:.1f}', xy=(smooth_eps[-1], end_val), 
                         xytext=(smooth_eps[-1], end_val + margin/2),
                         arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='green')
                         
            plt.title(f'Training Improvement: +{improvement:.1f} Points\n(Zoomed into Trend)', fontsize=16, fontweight='bold')

        plt.xlabel('Training Updates', fontsize=12)
        plt.ylabel('Average Reward (Rescaled)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig('chart_training_zoom.png', dpi=300)
        print("✅ Đã lưu biểu đồ Zoom: chart_training_zoom.png")
        print(f"👉 Hãy mở ảnh này lên, bạn sẽ thấy đường xanh đi lên!")
        plt.show()

    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == '__main__':
    plot_zoom_trend(LOG_FILE)