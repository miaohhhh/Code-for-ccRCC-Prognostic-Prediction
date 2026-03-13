import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 路径设置与文件夹创建
# ==========================================

data_path = r""
output_dir = r""

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 2. 数据读取
# ==========================================

df = pd.read_csv(data_path)
print(f"\n数据读取成功！")
print(f"数据集总行数: {df.shape[0]}")
print(f"数据集总列数: {df.shape[1]}")

missing_values = df.isnull().sum().sum()
print(f"数据集中缺失值总数: {missing_values}")

# ==========================================
# 3. 变量标记与分组
# ==========================================

id_var = df.columns[0]
outcome_vars = df.columns[1:4].tolist()
clinical_risk_models = df.columns[4:7].tolist()
clinical_vars = df.columns[7:16].tolist()
clinical_cat_vars = df.columns[7:13].tolist()
clinical_cont_vars = df.columns[13:16].tolist()
radiomics_vars = df.columns[16:].tolist()

# ==========================================
# 4. 信息打印确认
# ==========================================

print("\n" + "="*50)
print("变量分组统计:")
print("="*50)
print(f"\n1. ID 变量 (1个):")
print(f" {id_var}")
print(f"\n2. 结局变量 ({len(outcome_vars)}个):")
print(f" {outcome_vars}")
print(f"\n3. 临床模型风险分层 ({len(clinical_risk_models)}个):")
print(f" {clinical_risk_models}")
print(f"\n4. 临床特征 ({len(clinical_vars)}个):")
print(f" - 分类变量 ({len(clinical_cat_vars)}个): {clinical_cat_vars}")
print(f" - 连续变量 ({len(clinical_cont_vars)}个): {clinical_cont_vars}")
print(f"\n5. 影像组学特征 ({len(radiomics_vars)}个):")
print(f" 前5个特征示例: {radiomics_vars[:5]}")
