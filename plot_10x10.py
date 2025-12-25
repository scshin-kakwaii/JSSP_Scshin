import matplotlib.pyplot as plt
import re
import numpy as np
import os

# ==========================================
# CẤU HÌNH (CHỈNH SỬA TẠI ĐÂY)
# ==========================================
LOG_FILENAME = 'result_5x5.txt'

# 1. CẤU HÌNH ZOOM (Bạn muốn xem đoạn nào?)
# Nếu muốn xem toàn bộ: Để START = 0, END = 10000
# Nếu muốn soi kỹ đoạn cuối: Để START = 8000, END = 10000
X_START = 0      
X_END = 600    

# 2. CẤU HÌNH VẠCH CHIA TRỤC X (Ticks)
# Mỗi bao nhiêu episode thì hiện 1 số mốc? 
# (Số càng nhỏ thì trục X càng chi tiết, nhưng đừng nhỏ quá kẻo chữ đè lên nhau)
X_TICK_STEP = 500

# 3. KÍCH THƯỚC ẢNH (Quan trọng để giãn hình)
# Tăng số đầu tiên (Width) lên để kéo giãn trục X
FIG_SIZE = (20, 8)  # Rộng 20 inch, Cao 8 inch (Rất rộng)
# ==========================================

def parse_log_file(filename):
    if not os.path.exists(filename):
        print(f"❌ LỖI: Không tìm thấy file '{filename}'")
        return None, None

    print(f"Đang đọc file: {filename} ...")
    episodes = []
    makespans = []
    
    current_ep = 0
    val_pattern = re.compile(r'VALIDATION:\s+(\d+\.\d+)')
    ep_pattern = re.compile(r'Ep\s+(\d+)\s+\|')

    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            ep_match = ep_pattern.search(line)
            if ep_match:
                current_ep = int(ep_match.group(1))

            val_match = val_pattern.search(line)
            if val_match:
                makespan = float(val_match.group(1))
                if current_ep > 0:
                    episodes.append(current_ep)
                    makespans.append(makespan)

    return np.array(episodes), np.array(makespans)

def plot_zoomed(episodes, makespans):
    # Lọc dữ liệu theo khoảng Zoom cấu hình ở trên
    mask = (episodes >= X_START) & (episodes <= X_END)
    zoom_eps = episodes[mask]
    zoom_mks = makespans[mask]

    if len(zoom_eps) == 0:
        print(f"⚠️ Không có dữ liệu trong khoảng từ {X_START} đến {X_END}.")
        return

    # Tạo biểu đồ với kích thước RỘNG
    plt.figure(figsize=FIG_SIZE) 

    # Vẽ đường Makespan
    plt.plot(zoom_eps, zoom_mks, color='#D32F2F', linewidth=2, marker='o', markersize=4, label='Validation Makespan')

    # Tìm điểm thấp nhất trong khoảng Zoom này
    min_idx = np.argmin(zoom_mks)
    min_val = zoom_mks[min_idx]
    min_ep = zoom_eps[min_idx]

    # Đánh dấu điểm tốt nhất
    plt.scatter(min_ep, min_val, color='gold', s=200, edgecolors='black', zorder=5, marker='*')
    plt.annotate(f'Best in range: {min_val:.1f}\n(Ep {min_ep})', 
                 xy=(min_ep, min_val), 
                 xytext=(min_ep, min_val - (max(zoom_mks) - min_val)*0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='center', fontweight='bold', fontsize=12)

    # --- CẤU HÌNH TRỤC X (KEY FIX) ---
    # Tạo các mốc thời gian từ Start đến End với bước nhảy Step
    ticks = np.arange(X_START, X_END + 1, X_TICK_STEP)
    plt.xticks(ticks, rotation=45, fontsize=10) # Xoay 45 độ cho đỡ dính nhau
    plt.xlim(X_START, X_END)

    plt.title(f'Validation Makespan (Zoom: Ep {X_START} - {X_END})', fontsize=16, fontweight='bold')
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Makespan', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5, which='both')
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    
    output_file = f'chart_zoom_{X_START}_{X_END}.png'
    plt.savefig(output_file, dpi=300)
    print(f"✅ Đã vẽ xong! Ảnh siêu rộng đã lưu tại: {output_file}")
    plt.show()

if __name__ == '__main__':
    eps, mks = parse_log_file(LOG_FILENAME)
    if eps is not None:
        plot_zoomed(eps, mks)