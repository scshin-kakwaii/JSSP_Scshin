import numpy as np
import os
from uniform_instance_gen import uni_instance_gen

# --- CẤU HÌNH KÍCH THƯỚC BÀI TOÁN ---
# Bạn sửa 2 số này tùy theo bài bạn muốn chạy (6x6 hay 20x10)
j = 10  # Số Jobs
m = 10   # Số Machines

l = 1
h = 99
batch_size = 100 # Số lượng mẫu validation (nên để 20-100)
seed = 200

# Tạo folder DataGen nếu chưa có
if not os.path.exists('./DataGen'):
    os.makedirs('./DataGen')
    print("Đã tạo mới folder ./DataGen")

np.random.seed(seed)

print(f"Đang sinh dữ liệu {j}x{m}...")
data = np.array([uni_instance_gen(n_j=j, n_m=m, low=l, high=h) for _ in range(batch_size)])
print(f"Kích thước dữ liệu: {data.shape}")

# Lưu file vào trong folder DataGen
file_name = './DataGen/generatedData{}_{}_Seed{}.npy'.format(j, m, seed)
np.save(file_name, data)

print(f"✅ Đã lưu file thành công tại: {file_name}")