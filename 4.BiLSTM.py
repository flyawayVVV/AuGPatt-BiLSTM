import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import time
from sklearn.preprocessing import MinMaxScaler
from GSA_PSO import gsa_pso_algorithm, generate_individual

# 1. 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================
# 2. 数据读取与预处理
# ==========================================
print("Loading and preprocessing data...")
# 请确保文件名路径正确
data = pd.read_excel('Animas River_imfs.xlsx')

# A. 分离 Target 和 IMFs
raw_target = data['target'].values.reshape(-1, 1)
raw_imfs = data.drop(columns=['target']).values

print(f"Original IMFs shape: {raw_imfs.shape}")
print(f"Original Target shape: {raw_target.shape}")

# B. 数据归一化 (保留原有的预处理逻辑)
scaler_imfs = MinMaxScaler(feature_range=(0, 1))
imfs_scaled = scaler_imfs.fit_transform(raw_imfs)

scaler_target = MinMaxScaler(feature_range=(0, 1))
target_scaled = scaler_target.fit_transform(raw_target)

target_flat = target_scaled.flatten()

# ==========================================
# 3. 模型定义
# ==========================================

# [移除] Autoencoder 类已删除

class BiLSTM(nn.Module):
    # 【修改】：增加 dropout_rate 参数
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 标准 BiLSTM 层
        # 【修改】：传入 dropout 参数 (注意：dropout 仅在 num_layers > 1 时生效)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # out shape: (batch_size, seq_len, hidden_size * 2)
        out, _ = self.lstm(x, (h0, c0))

        # 直接取最后一个时间步 (Last Time Step) 的输出
        out = out[:, -1, :]

        out = self.fc(out)
        return torch.sigmoid(out) # 保证结果非负

def create_sequences(data, target, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i : (i + seq_length)]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# [移除] train_autoencoder 函数已删除

# 【修改】：增加 weight_decay 参数
def train_bilstm(model, train_data, train_labels, epochs, batch_size, learning_rate, weight_decay):
    criterion = nn.MSELoss()
    # 【修改】：在优化器中传入 weight_decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        batch_loss = 0
        for i in range(0, len(train_data) - batch_size, batch_size):
            batch = train_data[i : i + batch_size]
            labels = train_labels[i : i + batch_size]
            
            inputs = torch.tensor(batch, dtype=torch.float32).to(device)  
            targets = torch.tensor(labels, dtype=torch.float32).to(device)  
            targets = targets.view(-1, 1)  

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            
        # 减少打印频率，避免刷屏，每10轮打印一次
        if (epoch + 1) % 10 == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = epochs - (epoch + 1)
            remaining_seconds = avg_time_per_epoch * remaining_epochs
            m, s = divmod(remaining_seconds, 60)
            h, m = divmod(m, 60)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {batch_loss:.6f} | ETA: {int(h)}h {int(m)}m {int(s)}s")

# ==========================================
# 4. 主程序流程
# ==========================================

# --- [修改] 步骤 1: 构造混合输入 (直接拼接) ---
# 原始代码是: encoded_imfs + target
# 修改后为: imfs_scaled + target
integrated_input = np.hstack((imfs_scaled, target_scaled))
print(f"Integrated Input Shape for Pure BiLSTM: {integrated_input.shape}")
print("(Autoencoder removed, using raw scaled IMFs + Target)")

# --- 步骤 2: 用户自定义超参数（这里作为基准测试，参数可以和之前保持一致以便对比，或者重新寻优）
num_layers = 3
batch_size = 64
learning_rate = 0.0001
dropout_rate = 0.2
weight_decay = 0.00001
seq_length = 3

print("Using user define best_hyperparameters...")
best_hyperparameters = [num_layers, batch_size, learning_rate, dropout_rate, weight_decay, seq_length]

# 如果你想重新运行优化，取消下面注释
# best_hyperparameters, best_fitness = gsa_pso_algorithm(...) 

# --- 步骤 3: 使用最优参数进行最终训练 ---
best_num_layers, best_batch_size, best_learning_rate, best_dropout_rate, best_weight_decay, best_sequence_length = best_hyperparameters

best_num_epochs = 5000  # 建议根据实际需求调整，例如 5000

train_data, train_labels = create_sequences(integrated_input, target_flat, int(best_sequence_length))

train_size = int(len(train_data) * 0.8)
train_data_train = train_data[:train_size]
train_labels_train = train_labels[:train_size]
train_data_test = train_data[train_size:]
train_labels_test = train_labels[train_size:]

input_size_final = integrated_input.shape[1] # 自动适配拼接后的维度
hidden_size = 64
output_size = 1

# 【修改】：实例化时传入 best_dropout_rate
bilstm_final = BiLSTM(input_size_final, hidden_size, int(best_num_layers), output_size, best_dropout_rate)
bilstm_final.to(device)

print(f"Starting Final Training (Pure BiLSTM)...")
# 【修改】：训练时传入 best_weight_decay
train_bilstm(bilstm_final, train_data_train, train_labels_train, int(best_num_epochs), int(best_batch_size), best_learning_rate, best_weight_decay)

# --- 步骤 4: 预测与结果保存 ---
bilstm_final.eval()
test_inputs = torch.tensor(train_data_test, dtype=torch.float32).to(device)

with torch.no_grad():
    test_outputs_scaled = bilstm_final(test_inputs).detach().cpu().numpy()

predicted_real = scaler_target.inverse_transform(test_outputs_scaled)
observed_real = scaler_target.inverse_transform(train_labels_test.reshape(-1, 1))

save_path = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(save_path):
    os.makedirs(save_path)

observed_vs_predicted = pd.DataFrame({
    'observed': observed_real.flatten(),
    'predicted': predicted_real.flatten(),
})

# [修改] 更新保存的文件名，加上 _Pure_BiLSTM
excel_path = os.path.join(save_path, 'observed_vs_predicted_AnimasRiver_Pure_BiLSTM.xlsx')
observed_vs_predicted.to_excel(excel_path, index=False, engine='openpyxl')
print(f"Results saved to {excel_path}")

# [修改] 只保存 BiLSTM 模型，不再保存 AE
save_file_lstm = os.path.join(save_path, 'Pure_BiLSTM.pth')
with open(save_file_lstm, 'wb') as f:
    torch.save(bilstm_final.state_dict(), f)

print("Model (Pure BiLSTM) saved.")
