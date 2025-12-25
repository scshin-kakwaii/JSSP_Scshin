import matplotlib.pyplot as plt
import ast
import os
import numpy as np

# --- CẤU HÌNH ---
# Đường dẫn file log (Lấy từ code cũ của bạn)
LOG_FILE = 'code_1/L2D/vali_6_6_1_99.txt' 
# Đường dẫn lưu ảnh (Lấy từ code cũ của bạn)
SAVE_PATH = '/Users/shenchiashin/Downloads/CAPSTONE_3RD MODIFICATION/code_1/L2D/chart_combined_6x6.png'

# Độ mượt (số càng to càng phẳng, 5 là đẹp nhất)
SMOOTH_WINDOW = 5  

def plot_combined_smooth(filename):
    if not os.path.exists(filename):
        print(f"❌ Không tìm thấy file {filename}. Hãy kiểm tra lại đường dẫn.")
        return

    try:
        with open(filename, 'r') as f:
            content = f.read()
            data = ast.literal_eval(content)
            
        # Chuyển sang numpy array để xử lý toán học
        episodes = np.array([x[0] for x in data])
        makespans = np.array([x[1] for x in data])
        
        # 1. CHUYỂN ĐỔI TRỤC X (Validation Step = Episode / 100)
        # Để giống format hình bạn của bạn
        validation_steps = episodes / 100

        # 2. LÀM MƯỢT DỮ LIỆU (Smoothing)
        if len(makespans) >= SMOOTH_WINDOW:
            window = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
            # Tính trung bình trượt
            y_smooth = np.convolve(makespans, window, mode='valid')
            # Cắt bớt trục X cho khớp độ dài sau khi làm mượt
            x_smooth = validation_steps[SMOOTH_WINDOW-1:]
        else:
            y_smooth = makespans
            x_smooth = validation_steps

        # --- VẼ BIỂU ĐỒ ---
        plt.figure(figsize=(10, 8)) # Tỉ lệ khung hình vuông vắn
        
        # Vẽ đường đã làm mượt
        plt.plot(x_smooth, y_smooth, 
                 color='#1f77b4',      # Màu xanh dương chuẩn (Tab:blue) giống hình mẫu
                 marker='o',           # Chấm tròn tại các điểm
                 linestyle='-',        # Đường liền
                 linewidth=2,          # Độ dày vừa phải
                 markersize=6,         # Kích thước chấm
                 label=f'Validation Trend (Smoothed)')

        # 3. Đánh dấu điểm tốt nhất (Dựa trên dữ liệu gốc thực tế)
        min_val = np.min(makespans)
        min_idx = np.argmin(makespans)
        best_step = validation_steps[min_idx]
        
        # Ngôi sao vàng cho điểm tốt nhất
        plt.scatter(best_step, min_val, color='gold', s=200, edgecolors='black', zorder=10, marker='*', label='Best Model')
        plt.annotate(f'Best: {min_val:.1f}', xy=(best_step, min_val), 
                     xytext=(best_step, min_val - 30), # Đẩy chữ xuống dưới
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='center', fontweight='bold', fontsize=12)

        # 4. CẤU HÌNH FORMAT (Style Match)
        plt.grid(True, which='both', linestyle='--', alpha=0.7) # Lưới nét đứt
        plt.title('Validation Performance (6x6)', fontsize=18, fontweight='bold')
        plt.xlabel('Validation Step (x100 episodes)', fontsize=16)
        plt.ylabel('Makespan', fontsize=16)
        
        # Chữ số trên trục to rõ
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.legend(fontsize=14, loc='upper right', frameon=True)
        plt.tight_layout()
        
        # Lưu file
        plt.savefig(SAVE_PATH, dpi=300)
        print(f"✅ Đã lưu biểu đồ đẹp tại: {SAVE_PATH}")
        plt.show()

    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == '__main__':
    plot_combined_smooth(LOG_FILE)