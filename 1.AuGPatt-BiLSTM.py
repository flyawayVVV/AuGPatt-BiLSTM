import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import time
from sklearn.preprocessing import MinMaxScaler  # 新增：归一化工具
from GSA_PSO import gsa_pso_algorithm, generate_individual

# 1. 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================
# 2. 数据读取与预处理 (核心修改部分)
# ==========================================
print("Loading and preprocessing data...")
data = pd.read_excel('Animas River_imfs.xlsx')

# A. 分离 Target 和 IMFs
# 假设 Excel 中有一列叫 'target'，其余全是 IMFs
raw_target = data['target'].values.reshape(-1, 1)  # 变为 (N, 1)
raw_imfs = data.drop(columns=['target']).values    # 变为 (N, 17)

print(f"Original IMFs shape: {raw_imfs.shape}")
print(f"Original Target shape: {raw_target.shape}")

# B. 数据归一化 (使用 [0, 1] 范围，配合 ReLU 激活函数)
# 对 IMFs 归一化 (Autoencoder 输入)
scaler_imfs = MinMaxScaler(feature_range=(0, 1))
imfs_scaled = scaler_imfs.fit_transform(raw_imfs)

# 对 Target 归一化 (BiLSTM 目标)
scaler_target = MinMaxScaler(feature_range=(0, 1))
target_scaled = scaler_target.fit_transform(raw_target)

# 为了方便后续切片，准备一个展平的 target
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


class BiLSTM_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.0): # 修改：增加 dropout_rate 参数
        super(BiLSTM_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 修改：在 LSTM 中激活 dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, 1)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        attention_w = torch.tanh(self.attention_layer(out))
        attention_w = torch.softmax(attention_w, dim=1)

        out = attention_w * out
        out = torch.sum(out, axis=1)

        out = self.fc(out)
        return torch.sigmoid(out) # 保证结果非负

def create_sequences(data, target, seq_length):
    xs = []
    ys = []
    # data: (N, Features), target: (N,)
    for i in range(len(data) - seq_length):
        x = data[i : (i + seq_length)]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def train_autoencoder(model, data, epochs, batch_size, learning_rate):
    """注意：Autoencoder 不需要 label，它是无监督学习 (Input=Target)"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Start training Autoencoder...")
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(data) - batch_size, batch_size):
            batch = data[i : i + batch_size]
            inputs = torch.tensor(batch, dtype=torch.float32).to(device)
            # Autoencoder 目标是还原输入
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch+1) % 10 == 0:
            print(f"Autoencoder Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

def train_bilstm(model, train_data, train_labels, epochs, batch_size, learning_rate, weight_decay): # 修改：增加 weight_decay 参数
    criterion = nn.MSELoss()
    # 修改：在优化器中激活 weight_decay
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
        
        #每10个Epoch打印一次ETA
        if (epoch + 1) % 10 == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = epochs - (epoch + 1)
            remaining_seconds = avg_time_per_epoch * remaining_epochs
            m, s = divmod(remaining_seconds, 60)
            h, m = divmod(m, 60)
            print(f"  >> BiLSTM Epoch [{epoch+1}/{epochs}] Loss: {batch_loss:.6f} | ETA: {int(h)}h {int(m)}m {int(s)}s")

def fitness_function(hyperparameters):
    # 解包参数
    num_layers, batch_size, learning_rate, dropout_rate, weight_decay, sequence_length = hyperparameters
    num_layers = int(num_layers)
    batch_size = int(batch_size)
    sequence_length = int(sequence_length)

    # 这里的 epoch 指 PSO 一组粒子（一套超参数）的训练次数
    num_epochs = 125  #1000
    
    # 使用 integrated_input (AE特征 + 原始数据)
    train_data, train_labels = create_sequences(integrated_input, target_flat, sequence_length)
    
    train_size = int(len(train_data) * 0.8)
    # 分割训练集和测试集
    train_data_train = train_data[:train_size]
    train_labels_train = train_labels[:train_size]
    train_data_test = train_data[train_size:]
    train_labels_test = train_labels[train_size:]

    # 【注意】Input Size 变成了 17 (16个特征 + 1个原始数据)
    input_size = integrated_input.shape[1] 
    hidden_size = 64
    output_size = 1
    
    # 修改：传入 dropout_rate
    bilstm = BiLSTM_Attention(input_size, hidden_size, num_layers, output_size, dropout_rate)
    bilstm.to(device)

    # 训练 - 修改：传入 weight_decay
    train_bilstm(bilstm, train_data_train, train_labels_train, num_epochs, batch_size, learning_rate, weight_decay)

    # 评估
    bilstm.eval()
    criterion = nn.MSELoss()
    test_inputs = torch.tensor(train_data_test, dtype=torch.float32).to(device)
    test_targets = torch.tensor(train_labels_test, dtype=torch.float32).to(device).view(-1, 1)
    
    with torch.no_grad():
        test_outputs = bilstm(test_inputs)
        loss = criterion(test_outputs, test_targets).item()

    # 记录
    with open('gsa_pso_results.txt', 'a') as f:
        f.write("Params: " + str(hyperparameters) + " | Loss: " + str(loss) + "\n")

    return loss

# ==========================================
# 4. 主程序流程
# ==========================================

# --- 步骤 1: 训练 Autoencoder ---
input_size_ae = imfs_scaled.shape[1] # 应该是 17 (如果有17个IMF)
encoding_dim = 16
autoencoder = Autoencoder(input_size_ae, encoding_dim)
autoencoder.to(device)

autoencoder_epochs = 500
autoencoder_batch_size = 32
autoencoder_learning_rate = 1e-3

# 注意：这里只传入归一化后的 IMFs，不带 Target
train_autoencoder(autoencoder, imfs_scaled, autoencoder_epochs, autoencoder_batch_size, autoencoder_learning_rate)

# --- 步骤 2: 构造 BiLSTM 的混合输入 (Feature Fusion) ---
# 1. 获取编码特征 (N, 16)
encoded_imfs = autoencoder.encoder(torch.tensor(imfs_scaled, dtype=torch.float32).to(device)).detach().cpu().numpy()

# 2. 【关键】拼接：Autoencoder特征 + 归一化的原始 Target
# 形状变化: (N, 16) + (N, 1) -> (N, 17)
integrated_input = np.hstack((encoded_imfs, target_scaled))
#integrated_input = encoded_imfs   #输入数据不包括真实值
print(f"Integrated Input Shape for BiLSTM: {integrated_input.shape}")

# --- 步骤 3: 运行 GSA-PSO 优化 ---
num_agents = 10    #10
num_iterations = 20    #20

G0 = 10
alpha = 20

# 设置控制变量 (1为运行算法，0或其他值为使用固定参数)
WeatherRunGSA_PSO = 1

if WeatherRunGSA_PSO == 1:
    print("Starting GSA-PSO optimization...")
    # 运行优化算法
    best_hyperparameters = gsa_pso_algorithm(fitness_function, num_agents, num_iterations, G0, alpha)
else:
    print("Using user define best_hyperparameters...")
    # 使用预设的超参数
    best_hyperparameters = [9, 32, 0.0007427638269166199, 0.3307870505314098, 5.987462265443377e-07, 8]

print("Best Hyperparameters found:", best_hyperparameters)

# --- 步骤 4: 使用最优参数进行最终训练 ---
best_num_layers, best_batch_size, best_learning_rate, best_dropout_rate, best_weight_decay, best_sequence_length = best_hyperparameters

# 最终训练 epoch 增加
best_num_epochs = 5000  #5000 

# 重新生成数据 (使用最优的 sequence_length)
train_data, train_labels = create_sequences(integrated_input, target_flat, int(best_sequence_length))

train_size = int(len(train_data) * 0.8)
train_data_train = train_data[:train_size]
train_labels_train = train_labels[:train_size]
train_data_test = train_data[train_size:]
train_labels_test = train_labels[train_size:]

# 实例化最终模型
input_size_final = integrated_input.shape[1] # 17
hidden_size = 64
output_size = 1
# 修改：传入 best_dropout_rate
bilstm_final = BiLSTM_Attention(input_size_final, hidden_size, int(best_num_layers), output_size, best_dropout_rate)
bilstm_final.to(device)

print(f"Starting Final Training with Best Params (Epochs: {best_num_epochs})...")
# 修改：传入 best_weight_decay
train_bilstm(bilstm_final, train_data_train, train_labels_train, int(best_num_epochs), int(best_batch_size), best_learning_rate, best_weight_decay)

# --- 步骤 5: 预测与结果保存 (含反归一化) ---
bilstm_final.eval()
test_inputs = torch.tensor(train_data_test, dtype=torch.float32).to(device)

with torch.no_grad():
    # 预测结果 (归一化状态 [0,1])
    test_outputs_scaled = bilstm_final(test_inputs).detach().cpu().numpy()

# 【核心】反归一化：将 [0,1] 还原回真实流量值 (例如 2000 m3/s)
# 注意：train_labels_test 也是归一化的，需要变回真实值对比
predicted_real = scaler_target.inverse_transform(test_outputs_scaled)
observed_real = scaler_target.inverse_transform(train_labels_test.reshape(-1, 1))

# 保存结果
save_path = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(save_path):
    os.makedirs(save_path)

observed_vs_predicted = pd.DataFrame({
    'observed': observed_real.flatten(),
    'predicted': predicted_real.flatten(),
})

excel_path = os.path.join(save_path, 'observed_vs_predicted_AnimasRiver_AuGPatt_BiLSTM.xlsx')
observed_vs_predicted.to_excel(excel_path, index=False, engine='openpyxl')
print(f"Results saved to {excel_path}")

# 保存模型权重
# 1. 保存 Autoencoder
save_file_ae = os.path.join(save_path, 'Autoencoder_AuGPatt_BiLSTM.pth')
with open(save_file_ae, 'wb') as f:
    torch.save(autoencoder.state_dict(), f)

# 2. 保存 BiLSTM
save_file_lstm = os.path.join(save_path, 'AuGPatt_BiLSTM.pth')
with open(save_file_lstm, 'wb') as f:
    torch.save(bilstm_final.state_dict(), f)

print("Models saved.")
