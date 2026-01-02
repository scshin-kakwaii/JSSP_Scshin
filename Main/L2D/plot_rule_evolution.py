import matplotlib.pyplot as plt
import re
import numpy as np
import os

# --- CẤU HÌNH ---
LOG_FILE = 'rule_trace_final.txt'
# Tên các rule theo thứ tự index 0-7
RULE_NAMES = ['SPT', 'LPT', 'STPT', 'LTPT', 'LOR', 'MOR', 'LQNO', 'MQNO']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

def parse_rule_trace(filename):
    if not os.path.exists(filename):
        print(f"❌ Không tìm thấy file {filename}")
        return None

    episodes = []
    # Data structure: [Episode_Index][Rule_Index] = Count
    rule_counts = [] 
    
    current_ep_counts = np.zeros(8)
    current_ep = -1
    has_data = False

    with open(filename, 'r') as f:
        for line in f:
            # Bắt dòng bắt đầu Episode mới
            # Mẫu: === Episode 100 ===
            ep_match = re.search(r'=== Episode (\d+) ===', line)
            if ep_match:
                # Lưu dữ liệu của ep trước (nếu có)
                if current_ep != -1 and has_data:
                    episodes.append(current_ep)
                    # Normalize về phần trăm (%)
                    total = np.sum(current_ep_counts)
                    if total > 0:
                        rule_counts.append(current_ep_counts / total)
                    else:
                        rule_counts.append(current_ep_counts)
                
                # Reset cho ep mới
                current_ep = int(ep_match.group(1))
                current_ep_counts = np.zeros(8)
                has_data = False
                continue

            # Bắt dòng chọn Rule
            # Mẫu: Step 1: rule SPT (index 0)
            rule_match = re.search(r'rule \w+ \(index (\d+)\)', line)
            if rule_match:
                idx = int(rule_match.group(1))
                if 0 <= idx < 8:
                    current_ep_counts[idx] += 1
                    has_data = True

    # Lưu ep cuối cùng
    if current_ep != -1 and has_data:
        episodes.append(current_ep)
        total = np.sum(current_ep_counts)
        rule_counts.append(current_ep_counts / total)

    return np.array(episodes), np.array(rule_counts)

def plot_evolution(episodes, rule_data):
    if len(episodes) == 0:
        print("File rỗng hoặc sai định dạng!")
        return

    plt.figure(figsize=(12, 7))
    
    # Chuyển đổi data để vẽ Stackplot
    # rule_data đang là (N_Eps, 8), cần transpose thành (8, N_Eps)
    y_stack = rule_data.T 
    
    plt.stackplot(episodes, y_stack, labels=RULE_NAMES, colors=COLORS, alpha=0.85)
    
    plt.title(f'Sự tiến hóa chiến thuật (Rule Usage Evolution) - 6x6', fontsize=16, fontweight='bold')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Tỷ lệ sử dụng Rule', fontsize=12)
    plt.margins(0, 0) # Sát lề
    
    # Legend bên ngoài để đỡ che hình
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Dispatching Rules")
    
    plt.tight_layout()
    plt.savefig('chart_rule_evolution.png', dpi=300)
    print("✅ Đã lưu biểu đồ: chart_rule_evolution.png")
    plt.show()

if __name__ == '__main__':
    ep, data = parse_rule_trace(LOG_FILE)
    if ep is not None:
        print(f"Đã phân tích {len(ep)} mốc kiểm tra.")
        plot_evolution(ep, data)