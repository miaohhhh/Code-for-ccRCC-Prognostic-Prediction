import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
import warnings
from lifelines.utils import concordance_index
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimSun']
rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置参数
# ==========================================

time_col = 'DFS_time'
event_col = 'DFS_event'

model_list = ['GBS', 'Leibovich', 'UISS', 'KEYNOTE-564']

colors = {
    'GBS': '#ff7f0e',
    'Leibovich': '#d62728',
    'UISS': '#17becf',
    'KEYNOTE-564': '#9467bd'
}

# ==========================================
# 2. 加载数据与 GBS 模型
# ==========================================

print("\n正在加载数据与 GBS 模型...")

train_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "2_test_data_scaled.csv"))

features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = features_df['Final_Feature_Name'].tolist()

try:
    gbs_model = joblib.load(os.path.join(output_dir, "4_gbs_model.pkl"))
    print("✓ GBS 模型加载成功")
except Exception as e:
    print(f"✗ GBS 模型加载失败: {e}")
    gbs_model = None


# ==========================================
# 3. 定义模型评分获取函数
# ==========================================

def get_model_scores(df, model_name):
    scores = None

    if model_name == 'GBS':
        if gbs_model is None:
            return None
        risk_scores = gbs_model.predict(df[final_features])
        scores = risk_scores
    else:
        if model_name in df.columns:
            scores = df[model_name].values
        else:
            print(f" ✗ 错误: 数据中未找到列 '{model_name}'")

    return scores


# ==========================================
# 4. 计算 C 指数
# ==========================================

def calculate_c_index(df, model_name):
    scores = get_model_scores(df, model_name)
    if scores is None:
        return None, None

    T = df[time_col].values
    E = df[event_col].values

    c_index = concordance_index(T, -scores, E)

    n_bootstrap = 1000
    c_indices = []
    n_samples = len(T)

    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        T_boot = T[indices]
        E_boot = E[indices]
        scores_boot = scores[indices]
        try:
            c_boot = concordance_index(T_boot, -scores_boot, E_boot)
            c_indices.append(c_boot)
        except:
            continue

    if len(c_indices) > 0:
        se = np.std(c_indices)
    else:
        se = 0.0

    return c_index, se


# ==========================================
# 5. 计算所有模型的 C 指数
# ==========================================

print("\n计算训练集和测试集的 C 指数...")

results = []

for model in model_list:
    print(f"\n正在计算 {model} 的 C 指数...")

    train_c, train_se = calculate_c_index(train_df, model)
    if train_c is None:
        print(f" ✗ 训练集计算失败")
        continue

    test_c, test_se = calculate_c_index(test_df, model)

    results.append({
        'Model': model,
        'Train_C_index': train_c,
        'Train_SE': train_se,
        'Test_C_index': test_c,
        'Test_SE': test_se,
        'Gap': train_c - test_c
    })

    print(f" 训练集: {train_c:.4f} ± {train_se:.4f}")
    print(f" 测试集: {test_c:.4f} ± {test_se:.4f}")

summary_df = pd.DataFrame(results)

# ==========================================
# 6. 输出结果
# ==========================================

print("\n" + "=" * 60)
print("C 指数汇总结果")
print("=" * 60)
print("\n按测试集 C 指数排序:")
print(summary_df.to_string(index=False))

summary_path = os.path.join(output_dir, "7_GBS_vs_Clinical_C_Index.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n汇总表已保存: {summary_path}")

# ==========================================
# 7. 可视化对比
# ==========================================

print("\n生成可视化对比...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
plt.subplots_adjust(wspace=0.35, bottom=0.25, top=0.92)

order = model_list
summary_df_ordered = summary_df.set_index('Model').loc[order].reset_index()

x = np.arange(len(summary_df_ordered))
width = 0.35

train_errors = [row['Train_SE'] for _, row in summary_df_ordered.iterrows()]
train_colors = [colors[row['Model']] for _, row in summary_df_ordered.iterrows()]

bars1 = ax1.bar(x - width / 2, summary_df_ordered['Train_C_index'], width, yerr=train_errors, capsize=5,
                color=train_colors, edgecolor='black', linewidth=1.5)

ax1.set_xlabel('模型', fontsize=22, fontweight='bold', labelpad=15)
ax1.set_ylabel('C-index', fontsize=22, fontweight='bold', labelpad=15)
ax1.set_title('训练集', fontsize=26, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(summary_df_ordered['Model'], rotation=45, ha='right')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([0, 1.1])
ax1.tick_params(labelsize=20)

for bar, err in zip(bars1, train_errors):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., height + err + 0.03, f'{height:.3f}', ha='center', va='bottom',
             fontsize=20, fontweight='bold')

test_errors = [row['Test_SE'] for _, row in summary_df_ordered.iterrows()]
test_colors = [colors[row['Model']] for _, row in summary_df_ordered.iterrows()]

bars2 = ax2.bar(x + width / 2, summary_df_ordered['Test_C_index'], width, yerr=test_errors, capsize=5,
                color=test_colors, edgecolor='black', linewidth=1.5)

ax2.set_xlabel('模型', fontsize=22, fontweight='bold', labelpad=15)
ax2.set_ylabel('C-index', fontsize=22, fontweight='bold', labelpad=15)
ax2.set_title('测试集', fontsize=26, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(summary_df_ordered['Model'], rotation=45, ha='right')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([0, 1.1])
ax2.tick_params(labelsize=20)

for bar, err in zip(bars2, test_errors):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height + err + 0.03, f'{height:.3f}', ha='center', va='bottom',
             fontsize=20, fontweight='bold')

plt.tight_layout()
plot_path = os.path.join(output_dir, "7_GBS_vs_Clinical_C_Index.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"对比图已保存: {plot_path}")

# ==========================================
# 8. 生成详细报告
# ==========================================

print("\n生成详细报告...")

report_path = os.path.join(output_dir, "7_GBS_vs_Clinical_C_Index_Report.txt")

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("GBS 与临床模型 C 指数对比报告\n")
    f.write("=" * 80 + "\n\n")

    f.write("分析概述:\n")
    f.write("- 评估指标: C-index (Concordance Index) ± 标准误\n")
    f.write("- 评估模型: GBS, Leibovich, UISS, KEYNOTE-564\n\n")

    f.write("=" * 80 + "\n")
    f.write("结果对比 (GBS 在首位)\n")
    f.write("=" * 80 + "\n\n")

    f.write(summary_df_ordered.to_string(index=False))
    f.write("\n\n")

    f.write("=" * 80 + "\n")
    f.write("关键发现\n")
    f.write("=" * 80 + "\n\n")

    gbs_result = summary_df_ordered[summary_df_ordered['Model'] == 'GBS'].iloc[0]
    f.write(f"1. GBS 模型表现:\n")
    f.write(f" - 训练集 C-index: {gbs_result['Train_C_index']:.4f} ± {gbs_result['Train_SE']:.4f}\n")
    f.write(f" - 测试集 C-index: {gbs_result['Test_C_index']:.4f} ± {gbs_result['Test_SE']:.4f}\n")
    f.write(f" - 过拟合程度: {gbs_result['Gap']:.4f}\n\n")

    f.write(f"2. 所有模型测试集 C-index 排名:\n")
    for i, row in summary_df_ordered.iterrows():
        f.write(f" {i + 1}. {row['Model']}: {row['Test_C_index']:.4f} ± {row['Test_SE']:.4f}\n")
    f.write("\n")

    f.write(f"3. 过拟合程度 (训练集-测试集 C-index 差异):\n")
    f.write(" 差异越小，泛化能力越好:\n")
    summary_by_gap = summary_df_ordered.sort_values('Gap')
    for i, row in summary_by_gap.iterrows():
        f.write(f" - {row['Model']}: {row['Gap']:.4f}\n")
    f.write("\n")

    f.write("=" * 80 + "\n")
    f.write("临床意义与讨论\n")
    f.write("=" * 80 + "\n\n")

    f.write("1. C-index 解读:\n")
    f.write(" - C-index = 0.5: 随机预测\n")
    f.write(" - C-index = 1.0: 完美预测\n")
    f.write(" - 通常认为 C-index > 0.7 具有临床价值\n\n")

    f.write("2. GBS 与临床模型对比:\n")
    f.write(" - GBS 是基于机器学习的梯度提升生存模型\n")
    f.write(" - 临床模型基于传统风险因素评分\n")
    f.write(" - 对比可以评估机器学习模型的临床应用价值\n\n")

    f.write("3. 标准误的意义:\n")
    f.write(" - 表示 C-index 估计的精确度\n")
    f.write(" - 较小的标准误表示模型性能估计更可靠\n")
    f.write(" - 可用于计算 95% 置信区间: C-index ± 1.96 × SE\n\n")

print(f"详细报告已保存: {report_path}")
