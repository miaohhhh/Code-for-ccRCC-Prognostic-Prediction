import pandas as pd
import numpy as np
import os
from scipy import stats
from datetime import datetime

# ==========================================
# 1. 数据准备
# ==========================================

train_filename = "train_data.csv"
test_filename = "test_data.csv"

try:
    train_df = pd.read_csv(os.path.join(output_dir, train_filename))
    test_df = pd.read_csv(os.path.join(output_dir, test_filename))
    print(f"✓ 成功读取: {train_filename} 和 {test_filename}")
except FileNotFoundError:
    print(f"✗ 错误: 未找到文件 {train_filename} 或 {test_filename}")
    print(f" 请确认文件是否存在于: {output_dir}")
    print(f" 注意: 需要是'划分后、标准化前'的文件。")
    exit()

all_df = pd.concat([train_df, test_df], ignore_index=True)
print(f"数据概况: 全部 {len(all_df)} 人 | 训练 {len(train_df)} 人 | 测试 {len(test_df)} 人")

# ==========================================
# 2. 自动识别并构建统计变量列表
# ==========================================

print("\n正在识别变量 (基于原始数据列)...")

all_cols = all_df.columns.tolist()

outcome_candidates = all_cols[1:4]
clinical_cat_cols = all_cols[4:13]
clinical_cont_cols = all_cols[13:16]

variables_to_describe = []

for col in outcome_candidates:
    col_lower = col.lower()
    if 'time' in col_lower:
        variables_to_describe.append(('结局变量', col, 'continuous'))
    elif 'recurrence' in col_lower:
        variables_to_describe.append(('结局变量', col, 'categorical'))
    elif 'event' in col_lower or 'status' in col_lower:
        variables_to_describe.append(('结局变量', col, 'categorical'))

for col in clinical_cat_cols:
    if col in all_df.columns:
        variables_to_describe.append(('临床特征(分类)', col, 'categorical'))

for col in clinical_cont_cols:
    if col in all_df.columns:
        variables_to_describe.append(('临床特征(连续)', col, 'continuous'))

print(f"✓ 识别到 {len(variables_to_describe)} 个变量 (已排除影像组学)")


# ==========================================
# 3. 统计函数定义
# ==========================================

def describe_continuous(df, var):
    if var not in df.columns:
        return "N/A"
    data = df[var].dropna()
    if len(data) == 0:
        return "N/A"
    median = np.median(data)
    min_val = np.min(data)
    max_val = np.max(data)
    return f"{median:.2f} ({min_val:.2f}-{max_val:.2f})"


def describe_categorical(df, var):
    if var not in df.columns:
        return "N/A"
    total = len(df)
    counts = df[var].value_counts(dropna=False)
    percentages = (counts / total * 100).round(1)
    result = []
    for cat, cnt, pct in zip(counts.index, counts.values, percentages.values):
        cat_str = "缺失" if pd.isna(cat) else str(cat)
        result.append(f"{cat_str}: {int(cnt)} ({pct}%)")
    return "; ".join(result)


def test_significance(df1, df2, var, var_type):
    try:
        if var not in df1.columns or var not in df2.columns:
            return "-", "N/A"
        d1, d2 = df1[var].dropna(), df2[var].dropna()
        if len(d1) == 0 or len(d2) == 0:
            return "-", "N/A"
        if var_type == 'categorical':
            all_vals = list(d1) + list(d2)
            unique_vals = sorted(set(all_vals))
            contingency = [[sum(d1 == v), sum(d2 == v)] for v in unique_vals]
            chi2, p, _, _ = stats.chi2_contingency(contingency)
            return f"χ²={chi2:.2f}", f"P={p:.3f}"
        else:
            u_stat, p = stats.mannwhitneyu(d1, d2, alternative='two-sided')
            return f"U={u_stat:.0f}", f"P={p:.3f}"
    except:
        return "-", "Error"


# ==========================================
# 4. 生成统计表格
# ==========================================

print("\n正在计算统计值...")

results = []

for category, var, var_type in variables_to_describe:
    row = {
        '类别': category,
        '变量': var,
        '类型': '分类' if var_type == 'categorical' else '连续',
    }

    row['全部患者'] = describe_categorical(all_df, var) if var_type == 'categorical' else describe_continuous(all_df,
                                                                                                              var)
    row['训练集'] = describe_categorical(train_df, var) if var_type == 'categorical' else describe_continuous(train_df,
                                                                                                              var)
    row['测试集'] = describe_categorical(test_df, var) if var_type == 'categorical' else describe_continuous(test_df,
                                                                                                             var)

    stat, p_val = test_significance(train_df, test_df, var, var_type)
    row['统计量'] = stat
    row['P值'] = p_val

    sig = ""
    try:
        if p_val != "N/A":
            p = float(p_val.split('=')[1])
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
    except:
        pass
    row['显著性'] = sig

    results.append(row)

description_df = pd.DataFrame(results)
description_df = description_df[[
    '类别', '变量', '类型', '全部患者', '训练集', '测试集', '统计量', 'P值', '显著性'
]]

# ==========================================
# 5. 保存文件
# ==========================================

print("\n正在保存文件...")

base_filename = "5_Descriptive_Statistics_Raw"
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
saved_paths = []

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

attempts = 0
max_attempts = 3
while attempts < max_attempts:
    attempts += 1
    filename = f"{base_filename}.csv" if attempts == 1 else f"{base_filename}_{current_time}_尝试{attempts}.csv"
    save_path = os.path.join(output_dir, filename)
    try:
        description_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        if os.path.exists(save_path):
            saved_paths.append(save_path)
            print(f"✓ 保存成功: {filename}")
            break
    except PermissionError:
        if attempts == max_attempts:
            print(f"⚠ 保存失败: 文件被占用，请关闭 Excel。")
        pass
    except Exception as e:
        print(f"✗ 错误: {e}")
        break

# ==========================================
# 6. 打印并输出结果
# ==========================================

print("\n" + "=" * 100)
print("表: 全部患者、训练集、测试集基线特征对比 (基于原始数据)")
print("=" * 100)
print("说明:")
print(" 1. 数据来源: 划分后、标准化前的原始数据 (非 Z-score)")
print(" 2. 连续变量: 中位数 (最小值-最大值)")
print(" 3. 分类变量: 类别名称: 频数 (%)")
print(" 4. 显著性: *** P<0.001, ** P<0.01, * P<0.05")
print("-" * 100)

pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 140)
print(description_df.to_string(index=False))

if saved_paths:
    import subprocess

    if os.name == 'nt':
        subprocess.Popen(f'explorer "{output_dir}"')
