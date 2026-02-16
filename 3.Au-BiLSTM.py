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
data = pd.read_excel('Animas River_imfs.xlsx')

# A. 分离 Target 和 IMFs
raw_target = data['target'].values.reshape(-1, 1)
raw_imfs = data.drop(columns=['target']).values

print(f"Original IMFs shape: {raw_imfs.shape}")
print(f"Original Target shape: {raw_target.shape}")

# B. 数据归一化
scaler_imfs = MinMaxScaler(feature_range=(0, 1))
imfs_scaled = scaler_imfs.fit_transform(raw_imfs)

scaler_target = MinMaxScaler(feature_range=(0, 1))
target_scaled = scaler_target.fit_transform(raw_target)

target_flat = target_scaled.flatten()

# ==========================================
# 3. 模型定义
# ==========================================
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(Autoencoder, self).__init__()
        
        # 编码器：增加深度，确保有非线性特征提取能力
        # 结构：Input(15) -> Hidden(32) -> Encoded(16)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Tanh(),                # Tanh 适合处理有正负波动的特征（虽然输入归一化了，但中间层保持对称性较好）
            nn.Linear(32, encoding_dim),
            nn.Tanh()                 # 这里的激活函数让编码特征分布在 -1 到 1 之间 (或者用 ReLU 也行)
        )
        
        # 解码器：对称结构
        # 结构：Encoded(16) -> Hidden(32) -> Output(15)
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.Tanh(),
            nn.Linear(32, input_size),
            nn.Sigmoid(),             # 【关键修改】：必须用 Sigmoid 将输出强制限制在 [0, 1]，匹配 MinMaxScaler
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 标准 BiLSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # out shape: (batch_size, seq_len, hidden_size * 2)
        out, _ = self.lstm(x, (h0, c0))

        # 【核心差异】：直接取最后一个时间步 (Last Time Step) 的输出
        # out[:, -1, :] shape: (batch_size, hidden_size * 2)
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

def train_autoencoder(model, data, epochs, batch_size, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Start training Autoencoder...")
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(data) - batch_size, batch_size):
            batch = data[i : i + batch_size]
            inputs = torch.tensor(batch, dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch+1) % 10 == 0:
            print(f"Autoencoder Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# 【修改点3】：函数签名增加 weight_decay 参数
def train_bilstm(model, train_data, train_labels, epochs, batch_size, learning_rate, weight_decay):
    criterion = nn.MSELoss()
    # 【修改点4】：在优化器中传入 weight_decay
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
            
        #print(f"Epoch [{epoch+1}/{epochs}], Loss: {batch_loss:.6f}")
        
        if (epoch + 1) % 10 == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = epochs - (epoch + 1)
            remaining_seconds = avg_time_per_epoch * remaining_epochs
            m, s = divmod(remaining_seconds, 60)
            h, m = divmod(m, 60)
            print(f"  >> BiLSTM Epoch [{epoch+1}/{epochs}] Loss: {batch_loss:.6f} | ETA: {int(h)}h {int(m)}m {int(s)}s")


# ==========================================
# 4. 主程序流程
# ==========================================

# --- 步骤 1: 训练 Autoencoder ---
input_size_ae = imfs_scaled.shape[1]
encoding_dim = 16
autoencoder = Autoencoder(input_size_ae, encoding_dim)
autoencoder.to(device)

autoencoder_epochs = 500
autoencoder_batch_size = 32
autoencoder_learning_rate = 1e-3

train_autoencoder(autoencoder, imfs_scaled, autoencoder_epochs, autoencoder_batch_size, autoencoder_learning_rate)

# --- 步骤 2: 构造混合输入 ---
encoded_imfs = autoencoder.encoder(torch.tensor(imfs_scaled, dtype=torch.float32).to(device)).detach().cpu().numpy()
integrated_input = np.hstack((encoded_imfs, target_scaled))
#integrated_input = encoded_imfs   #输入数据不包括真实值
print(f"Integrated Input Shape for BiLSTM: {integrated_input.shape}")

# --- 步骤 3: 用户自定义超参数（基于经验或文献）
num_layers = 3
batch_size = 64
learning_rate = 0.0001
dropout_rate = 0.2
weight_decay = 0.00001
seq_length = 3

print("Using user define best_hyperparameters...")
best_hyperparameters = [num_layers, batch_size, learning_rate, dropout_rate, weight_decay, seq_length]

# --- 步骤 4: 使用最优参数进行最终训练 ---
best_num_layers, best_batch_size, best_learning_rate, best_dropout_rate, best_weight_decay, best_sequence_length = best_hyperparameters

best_num_epochs = 5000  #5000

train_data, train_labels = create_sequences(integrated_input, target_flat, int(best_sequence_length))

train_size = int(len(train_data) * 0.8)
train_data_train = train_data[:train_size]
train_labels_train = train_labels[:train_size]
train_data_test = train_data[train_size:]
train_labels_test = train_labels[train_size:]

input_size_final = integrated_input.shape[1]
hidden_size = 64
output_size = 1

# 【修改】：实例化 BiLSTM
# 【修改点5】：传入 best_dropout_rate
bilstm_final = BiLSTM(input_size_final, hidden_size, int(best_num_layers), output_size, best_dropout_rate)
bilstm_final.to(device)

print(f"Starting Final Training with Best Params (Standard BiLSTM)...")
# 【修改点6】：传入 best_weight_decay
train_bilstm(bilstm_final, train_data_train, train_labels_train, int(best_num_epochs), int(best_batch_size), best_learning_rate, best_weight_decay)

# --- 步骤 5: 预测与结果保存 ---
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

excel_path = os.path.join(save_path, 'observed_vs_predicted_AnimasRiver_Au_BiLSTM.xlsx')
observed_vs_predicted.to_excel(excel_path, index=False, engine='openpyxl')
print(f"Results saved to {excel_path}")

# 保存模型权重
save_file_ae = os.path.join(save_path, 'Autoencoder_Au_BiLSTM.pth')
with open(save_file_ae, 'wb') as f:
    torch.save(autoencoder.state_dict(), f)

save_file_lstm = os.path.join(save_path, 'Au_BiLSTM.pth')
with open(save_file_lstm, 'wb') as f:
    torch.save(bilstm_final.state_dict(), f)

print("Models (Standard BiLSTM) saved.")
