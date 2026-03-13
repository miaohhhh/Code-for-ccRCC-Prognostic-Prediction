import pandas as pd
import numpy as np
import os
import warnings
from scipy import stats
import matplotlib.pyplot as plt

try:
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from lifelines.utils import concordance_index

    SKSURV_AVAILABLE = True
except ImportError:
    print("错误: 请安装 scikit-survival 库")
    SKSURV_AVAILABLE = False

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置区域
# ==========================================

N_BOOTSTRAPS = 100
BASE_RANDOM_STATE = 42

time_col = 'DFS_time'
event_col = 'DFS_event'
GBS_PARAMS = {
    'learning_rate': 0.01,
    'max_depth': 3,
    'min_samples_split': 10,
    'n_estimators': 200,
    'subsample': 0.7
}

# ==========================================
# 2. 加载数据
# ==========================================

print("\n加载数据...")

train_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "2_test_data_scaled.csv"))

features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = features_df['Final_Feature_Name'].tolist()


def get_structured_y(df):
    return np.array([(df.loc[i, event_col], df.loc[i, time_col]) for i in df.index],
                    dtype=[('event', '?'), ('time', '<f8')])


X_test = test_df[final_features].values
y_test_struct = get_structured_y(test_df)
T_test = test_df[time_col].values
E_test = test_df[event_col].values

X_train_full = train_df[final_features].values
y_train_struct_full = get_structured_y(train_df)


# ==========================================
# 3. 函数定义
# ==========================================

def calculate_icc_single(data):
    data_clean = data.dropna()
    if len(data_clean) < 2:
        return np.nan

    k = data_clean.shape[1]
    n = data_clean.shape[0]

    grand_mean = data_clean.values.mean()
    row_means = data_clean.mean(axis=1).values
    col_means = data_clean.mean(axis=0).values

    SS_total = ((data_clean.values - grand_mean) ** 2).sum()
    SS_rows = k * ((row_means - grand_mean) ** 2).sum()
    SS_cols = n * ((col_means - grand_mean) ** 2).sum()
    SS_error = SS_total - SS_rows - SS_cols

    MS_rows = SS_rows / (n - 1) if n > 1 else 0
    MS_error = SS_error / ((n - 1) * (k - 1)) if (n > 1 and k > 1) else 0

    if MS_rows + (k - 1) * MS_error == 0:
        return np.nan

    icc = (MS_rows - MS_error) / (MS_rows + (k - 1) * MS_error)
    return icc


# ==========================================
# 4. 循环训练与预测
# ==========================================

print(f"\n开始进行 {N_BOOTSTRAPS} 次重采样训练...")
print("注意：n_estimators=200 且 learning_rate=0.01，训练可能需要几分钟时间。\n")

predictions_cols = [f'Iter_{i}' for i in range(N_BOOTSTRAPS)]
predictions_df = pd.DataFrame(index=test_df.index, columns=predictions_cols)
c_index_list = []

if not SKSURV_AVAILABLE:
    print("程序终止：缺少必要的库。")
else:
    for i in range(N_BOOTSTRAPS):
        print(f" 正在运行第 {i + 1}/{N_BOOTSTRAPS} 次...", end=" ")

        current_seed = BASE_RANDOM_STATE + i

        sample_indices = np.random.choice(
            len(X_train_full), size=len(X_train_full), replace=True
        )
        X_train_bs = X_train_full[sample_indices]
        y_train_bs = y_train_struct_full[sample_indices]

        current_params = GBS_PARAMS.copy()
        current_params['random_state'] = current_seed

        model = GradientBoostingSurvivalAnalysis(**current_params)
        model.fit(X_train_bs, y_train_bs)

        pred_risk = model.predict(X_test)
        c_val = concordance_index(T_test, -pred_risk, E_test)
        c_index_list.append(c_val)
        predictions_df[predictions_cols[i]] = pred_risk

        print(f"C-index: {c_val:.4f}")

# ==========================================
# 5. 计算 ICC 与 统计结果
# ==========================================

print("\n" + "=" * 60)
print("稳定性分析结果")
print("=" * 60)

icc_value = calculate_icc_single(predictions_df)
c_mean = np.mean(c_index_list)
c_std = np.std(c_index_list)

print(f"\n1. 模型预测一致性")
print(f" ICC 值: {icc_value:.4f}")

if icc_value > 0.9:
    print(" 评价: 极高稳定性，模型预测非常一致")
elif icc_value > 0.75:
    print(" 评价: 良好稳定性")
elif icc_value > 0.5:
    print(" 评价: 中等稳定性")
else:
    print(" 评价: 稳定性较差，模型对数据扰动敏感")

print(f"\n2. 模型性能波动 on Test Set:")
print(f" 平均 C-index: {c_mean:.4f}")
print(f" 标准差 (SD): {c_std:.4f}")
print(f" 95% CI: [{c_mean - 1.96 * c_std:.4f}, {c_mean + 1.96 * c_std:.4f}]")

report_path = os.path.join(output_dir, "15_GBS_Stability_ICC_Report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("GBS 模型稳定性评估报告\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"模型超参数:\n{GBS_PARAMS}\n\n")
    f.write(f"Bootstrap 次数: {N_BOOTSTRAPS}\n")
    f.write(f"评估方法: 对固定测试集进行多次重采样训练，计算预测分数的 ICC\n\n")
    f.write(f"1. 预测一致性: {icc_value:.4f}\n")
    f.write(f"2. C-index 统计:\n")
    f.write(f" Mean: {c_mean:.4f}\n")
    f.write(f" SD: {c_std:.4f}\n")
    f.write(f" Min: {np.min(c_index_list):.4f}\n")
    f.write(f" Max: {np.max(c_index_list):.4f}\n")

pred_save_path = os.path.join(output_dir, "15_GBS_Bootstrap_Predictions.csv")
predictions_df.to_csv(pred_save_path)

print(f"\n详细报告已保存: {report_path}")
print(f"预测分数矩阵已保存: {pred_save_path}")

# ==========================================
# 6. 可视化：预测分数的分布与波动
# ==========================================

print("\n生成可视化图表...")

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
plt.subplots_adjust(wspace=0.35, bottom=0.25, top=0.92)

np.random.seed(0)
sample_indices_to_plot = np.random.choice(len(predictions_df), size=min(5, len(predictions_df)), replace=False)
plot_data = predictions_df.iloc[sample_indices_to_plot].T
plot_data.plot(kind='box', ax=axes[0])

axes[0].set_title("预测风险分数波动示例", fontsize=26, fontweight='bold', pad=20)
axes[0].set_ylabel('预测风险分数', fontsize=22, fontweight='bold', labelpad=15)
axes[0].set_xlabel('患者编号 (测试集)', fontsize=22, fontweight='bold', labelpad=15)
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(labelsize=20)

axes[1].plot(range(1, N_BOOTSTRAPS + 1), c_index_list, marker='o', linestyle='-', color='#1f77b4', markersize=4)
axes[1].axhline(y=c_mean, color='r', linestyle='--', linewidth=2, label=f'均值 ({c_mean:.4f})')
axes[1].fill_between(range(1, N_BOOTSTRAPS + 1), c_mean - c_std, c_mean + c_std, color='r', alpha=0.1,
                     label=f'±1 标准差')

axes[1].set_title("测试集 C-index 波动趋势", fontsize=26, fontweight='bold', pad=20)
axes[1].set_xlabel('迭代次数', fontsize=22, fontweight='bold', labelpad=15)
axes[1].set_ylabel('C-index', fontsize=22, fontweight='bold', labelpad=15)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(labelsize=20)
axes[1].set_ylim([max(0.5, c_mean - 0.1), min(1.0, c_mean + 0.1)])

plt.tight_layout()
plot_save_path = os.path.join(output_dir, "15_GBS_Stability_Analysis.png")
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"图表已保存: {plot_save_path}")

print("\n" + "=" * 60)
print("稳定性评估完成！")
print("=" * 60)
