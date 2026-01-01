import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Import các module từ code của bạn
from Params import configs
from JSSP_Env import SJSSP
from PPO_jssp_multiInstances import PPO
from mb_agg import g_pool_cal
from agent_utils import greedy_select_action

# ==========================================
# CẤU HÌNH (SỬA TẠI ĐÂY)
# ==========================================
# 1. Kích thước bài toán (Phải khớp với Model đã train)
N_J = 6
N_M = 6

# 2. Tên file Model (File .pth tốt nhất bạn đã lưu)
MODEL_FILE = '6_6_Best.pth' 

# 3. File dữ liệu (Đảm bảo file này tồn tại trong folder DataGen)
DATA_FILE = f'DataGen/generatedData{N_J}_{N_M}_Seed200.npy'

# 4. Chọn bài toán số mấy để giải? (0 là bài đầu tiên)
INSTANCE_INDEX = 0 

# 5. Tên file xuất ra (luôn lưu cùng folder bạn chạy script)
OUTPUT_TXT = str(Path.cwd() / 'schedule_result.txt')
OUTPUT_IMG = str(Path.cwd() / 'gantt_result.png')
# ==========================================

device = torch.device(configs.device)

def solve_and_extract():
    # --- 1. LOAD DỮ LIỆU ---
    data_path = Path(__file__).parent / DATA_FILE
    if not data_path.exists():
        print(f"❌ LỖI: Không tìm thấy file dữ liệu tại {data_path}")
        return None, 0

    print(f"📖 Đang đọc dữ liệu từ: {data_path}")
    full_dataset = np.load(data_path, allow_pickle=True)
    # Lấy instance cụ thể
    instance_data = (full_dataset[INSTANCE_INDEX][0], full_dataset[INSTANCE_INDEX][1])

    # --- 2. LOAD MODEL ---
    print(f"🧠 Đang load model: {MODEL_FILE}")
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=N_J, n_m=N_M,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim, hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)
    
    model_path = Path(__file__).parent / MODEL_FILE
    try:
        ppo.policy.load_state_dict(torch.load(model_path, map_location=device))
        ppo.policy.eval()
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file model {model_path}")
        return None, 0

    # --- 3. CHẠY GIẢI (INFERENCE) ---
    print("🚀 Đang xếp lịch...")
    env = SJSSP(n_j=N_J, n_m=N_M)
    obs, _ = env.reset(options={'data': instance_data})
    
    g_pool_step = g_pool_cal(configs.graph_pool_type, 
                             torch.Size([1, env.number_of_tasks, env.number_of_tasks]), 
                             env.number_of_tasks, device)

    done = False
    while not done:
        fea = torch.from_numpy(obs['fea']).to(device)
        adj = torch.from_numpy(obs['adj']).to(device).to_sparse()
        cand = torch.from_numpy(obs['candidate']).to(device).unsqueeze(0)
        mask = torch.from_numpy(obs['mask']).to(device).unsqueeze(0)
        rule_fea = torch.from_numpy(obs['rule_features']).to(device).unsqueeze(0)

        with torch.no_grad():
            pi, _ = ppo.policy(fea, g_pool_step, None, adj, cand, mask, rule_fea)
        
        # Chọn hành động tốt nhất (Greedy)
        action = greedy_select_action(pi)
        obs, _, done, _, _ = env.step(action.item())

    makespan = env.max_endTime
    print(f"✅ Hoàn thành! Makespan = {makespan}")

    # --- 4. TRÍCH XUẤT LỊCH TRÌNH ---
    schedule = []
    for m_id in range(N_M):
        op_ids = env.opIDsOnMchs[m_id]
        start_times = env.mchsStartTimes[m_id]
        
        # Lọc các giá trị hợp lệ (bỏ padding âm)
        valid_indices = np.where(op_ids >= 0)[0]
        
        for idx in valid_indices:
            op_id = op_ids[idx]
            start = start_times[idx]
            job_id = op_id // N_M
            op_in_job = op_id % N_M
            duration = env.dur[job_id, op_in_job]
            
            schedule.append({
                'Machine': m_id + 1, # Để in ra là Machine 1, 2...
                'Job': job_id + 1,   # Job 1, 2...
                'Op_ID': op_id,
                'Start': start,
                'Duration': duration,
                'End': start + duration
            })
            
    return schedule, makespan

def save_to_txt(schedule, makespan):
    # Sắp xếp theo Máy và Thời gian bắt đầu
    schedule.sort(key=lambda x: (x['Machine'], x['Start']))
    
    with open(OUTPUT_TXT, 'w') as f:
        f.write(f"Scheduling Result for {N_J}x{N_M} Instance\n")
        f.write(f"Total Makespan: {makespan}\n")
        f.write("="*60 + "\n")
        f.write(f"{'Machine':<10} | {'Job':<10} | {'Start':<10} | {'End':<10} | {'Duration':<10}\n")
        f.write("-" * 60 + "\n")
        
        for item in schedule:
            f.write(f"M{item['Machine']:<9} | J{item['Job']:<9} | {item['Start']:<10} | {item['End']:<10} | {item['Duration']:<10}\n")
            
    print(f"📄 Đã lưu chi tiết lịch trình vào: {OUTPUT_TXT}")

def plot_gantt(schedule, makespan):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bảng màu cho các Job
    colors = plt.cm.tab20(np.linspace(0, 1, N_J))
    
    for item in schedule:
        m_idx = item['Machine'] - 1 # Về lại index 0 để vẽ
        j_idx = item['Job'] - 1
        start = item['Start']
        dur = item['Duration']
        
        # Vẽ Block
        rect = patches.Rectangle((start, m_idx - 0.4), dur, 0.8, 
                                 linewidth=1, edgecolor='black', facecolor=colors[j_idx % 20])
        ax.add_patch(rect)
        
        # Ghi tên Job (J1, J2...)
        if dur > 2: # Chỉ ghi nếu ô đủ rộng
            ax.text(start + dur/2, m_idx, f'J{item["Job"]}', 
                    ha='center', va='center', color='white', fontweight='bold', fontsize=9)

    # Trang trí trục
    ax.set_yticks(range(N_M))
    ax.set_yticklabels([f'Machine {i+1}' for i in range(N_M)], fontsize=12, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_title(f'Gantt Chart - Makespan: {makespan}', fontsize=16, fontweight='bold')
    
    ax.set_xlim(0, makespan * 1.05)
    ax.set_ylim(-0.5, N_M - 0.5)
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"🖼️  Đã lưu biểu đồ Gantt vào: {OUTPUT_IMG}")
    plt.show()

if __name__ == '__main__':
    schedule_data, final_makespan = solve_and_extract()
    
    if schedule_data:
        save_to_txt(schedule_data, final_makespan)
        plot_gantt(schedule_data, final_makespan)