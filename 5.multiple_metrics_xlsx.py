#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# ==========================================
# 1. 定义指标计算函数 (基于提供的公式)
# ==========================================

def calc_rmse(obs, pred):
    """
    (1) RMSE: Root Mean Square Error
    Range: 0 (best) to ∞
    """
    return np.sqrt(np.mean((obs - pred) ** 2))

def calc_mae(obs, pred):
    """
    (2) MAE: Mean Absolute Error
    Range: 0 (best) to ∞
    """
    return np.mean(np.abs(obs - pred))

def calc_mare(obs, pred):
    """
    (3) MARE: Mean Absolute Relative Error
    Formula: 1/N * sum( |(obs - pred) / obs| )
    注意: 如果观测值为0，该指标会趋向无穷大。这里我们将排除 obs=0 的点，
    或者加上极小值 epsilon，通常水文中建议排除 0 值计算 MARE。
    这里采用排除 obs=0 的数据点的策略。
    """
    # 筛选出 obs 不为 0 的索引
    valid_mask = obs != 0
    if np.sum(valid_mask) == 0:
        return np.nan # 如果全是0，无法计算
    
    o_filtered = obs[valid_mask]
    p_filtered = pred[valid_mask]
    
    return np.mean(np.abs((o_filtered - p_filtered) / o_filtered))

def calc_nse(obs, pred):
    """
    (4) NSE: Nash-Sutcliffe Efficiency
    Range: -∞ to 1 (best)
    """
    numerator = np.sum((obs - pred) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    if denominator == 0:
        return np.nan
    return 1 - (numerator / denominator)

def calc_r(obs, pred):
    """
    (5) R: Pearson Correlation Coefficient
    Range: -1 to 1 (best)
    """
    if len(obs) < 2:
        return np.nan
    return np.corrcoef(obs, pred)[0, 1]

def calc_kge(obs, pred):
    """
    (6) KGE: Kling-Gupta Efficiency (2012 version based on gamma/beta symbols)
    Formula: 1 - sqrt( (r-1)^2 + (gamma-1)^2 + (beta-1)^2 )
    其中:
    r = Pearson correlation coefficient
    beta = mean(pred) / mean(obs)  (Bias ratio)
    gamma = (CV_pred / CV_obs) = (std_pred/mean_pred) / (std_obs/mean_obs) (Variability ratio)
    """
    mean_obs = np.mean(obs)
    mean_pred = np.mean(pred)
    
    if mean_obs == 0 or mean_pred == 0:
        return np.nan # 避免除以0

    std_obs = np.std(obs)
    std_pred = np.std(pred)
    
    r = np.corrcoef(obs, pred)[0, 1]
    
    beta = mean_pred / mean_obs
    
    cv_obs = std_obs / mean_obs
    cv_pred = std_pred / mean_pred
    gamma = cv_pred / cv_obs
    
    kge = 1 - np.sqrt((r - 1)**2 + (gamma - 1)**2 + (beta - 1)**2)
    return kge

# 封装一个函数来批量计算一组数据的6个指标
def calculate_all_metrics(obs, pred):
    return {
        "RMSE": calc_rmse(obs, pred),
        "MAE": calc_mae(obs, pred),
        "MARE": calc_mare(obs, pred),
        "NSE": calc_nse(obs, pred),
        "R": calc_r(obs, pred),
        "KGE": calc_kge(obs, pred)
    }

# ==========================================
# 2. 主程序逻辑
# ==========================================

input_filename = '4.observed_vs_predicted_AnimasRiver_Pure_BiLSTM.xlsx'
output_filename = '4.metrics_Pure_BiLSTM.xlsx'

print(f"正在读取数据: {input_filename} ...")
try:
    df = pd.read_excel(input_filename)
except FileNotFoundError:
    print(f"错误: 找不到文件 {input_filename}")
    exit()

# 数据清洗：确保 'observed' 和 'predicted' 列存在且为数值
# 将非数值转换为 NaN 并删除
df['observed'] = pd.to_numeric(df['observed'], errors='coerce')
df['predicted'] = pd.to_numeric(df['predicted'], errors='coerce')
df.dropna(subset=['observed', 'predicted'], inplace=True)

# 转换为 numpy 数组以便计算
all_obs = df['observed'].values
all_pred = df['predicted'].values

print(f"有效数据行数: {len(all_obs)}")

# ==========================================
# 3. 数据集划分 (全部, 前20%, 后20%)
# ==========================================

# 为了切分高流量和低流量，我们需要基于 Observed 值对数据进行排序
# 创建一个临时 DataFrame 用于排序
temp_df = pd.DataFrame({'obs': all_obs, 'pred': all_pred})

# 总数量
N = len(temp_df)
top_n = int(N * 0.2) # 20% 的数据量

# --- A. 全部数据 ---
metrics_all = calculate_all_metrics(temp_df['obs'].values, temp_df['pred'].values)

# --- B. 高流量 (Top 20%) ---
# 按 obs 降序排列 (大到小)，取前 20%
df_high = temp_df.sort_values(by='obs', ascending=False).head(top_n)
metrics_high = calculate_all_metrics(df_high['obs'].values, df_high['pred'].values)

# --- C. 低流量 (Bottom 20%) ---
# 按 obs 升序排列 (小到大)，取前 20%
df_low = temp_df.sort_values(by='obs', ascending=True).head(top_n)
metrics_low = calculate_all_metrics(df_low['obs'].values, df_low['pred'].values)

# ==========================================
# 4. 结果汇总与保存
# ==========================================

# 定义指标顺序
metric_names = ["RMSE", "MAE", "MARE", "NSE", "R", "KGE"]

# 准备保存的数据结构
data_to_save = {
    'Metric': metric_names,
    'All_Data': [metrics_all[m] for m in metric_names],
    'High_Flow_Top20%': [metrics_high[m] for m in metric_names],
    'Low_Flow_Bottom20%': [metrics_low[m] for m in metric_names]
}

results_df = pd.DataFrame(data_to_save)

print("\n计算结果预览:")
print(results_df)

try:
    results_df.to_excel(output_filename, index=False)
    print(f"\n成功! 结果已保存至文件: {output_filename}")
    print("包含了：全部数据、高流量(前20%)、低流量(后20%) 三列精度指标。")
except Exception as e:
    print(f"\n保存文件失败: {e}")
