import pandas as pd
from PyEMD import CEEMDAN
from statsmodels.tsa.ar_model import AutoReg
import numpy as np
import os

# ================= 配置区域 =================
# 输入文件名 (必须是 .xlsx 格式，且在当前脚本同一目录下)
INPUT_FILENAME = '09361500-Animas River.xlsx' 

# 输出文件名
OUTPUT_FILENAME = 'Animas Rive_imfs.xlsx'

# Excel中存放径流数据的列名
DATA_COL_NAME = 'Data' 
# ===========================================

def find_optimal_lag_using_bic(data, max_lag):
    bic_values = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        model = AutoReg(data, lags=lag)
        results = model.fit()
        bic_values[lag - 1] = results.bic
    return np.argmin(bic_values) + 1

# 【重点修改】将执行代码放入 main 保护块中
if __name__ == '__main__':
    # 获取当前脚本所在的绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接完整路径
    input_path = os.path.join(base_dir, INPUT_FILENAME)
    output_path = os.path.join(base_dir, OUTPUT_FILENAME)

    print(f"Reading data from: {input_path}")

    # 读取数据 (强制使用 openpyxl 引擎读取 .xlsx)
    try:
        df = pd.read_excel(input_path, engine='openpyxl')
        data = df[DATA_COL_NAME].values
    except FileNotFoundError:
        print(f"Error: 找不到文件 {input_path}，请检查路径。")
        exit()
    except KeyError:
        print(f"Error: Excel中找不到列名 '{DATA_COL_NAME}'，请检查配置。")
        exit()

    # 计算最优滞后 (仅用于展示信息)
    max_lag = 20
    # 注意：如果数据量很大，find_optimal_lag_using_bic 也会有点慢，但通常没事
    optimal_lag = find_optimal_lag_using_bic(data, max_lag)
    print(f"Optimal lag: {optimal_lag}")

    print("Starting CEEMDAN decomposition... (This may take a while)")
    
    # Apply CEEMDAN
    # 在 Windows 下，这行代码及其调用必须在 if __name__ == '__main__': 内部
    ceemdan = CEEMDAN()
    
    # 你可以通过设置 processes 参数来控制并行数量，或者设为 1 关闭并行（如果还是报错的话）
    # ceemdan = CEEMDAN(processes=1) 
    
    imfs = ceemdan(data)
    print("Decomposition finished.")

    # 整理结果
    imfs_df = pd.DataFrame(imfs.T, columns=[f'IMF{i+1}' for i in range(imfs.shape[0])])

    # 【关键】将原始数据作为 'target' 列保存，防止后续训练代码报错
    imfs_df['target'] = data

    # 保存结果
    imfs_df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"File saved successfully to: {output_path}")
