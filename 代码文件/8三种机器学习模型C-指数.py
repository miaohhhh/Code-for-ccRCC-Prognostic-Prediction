import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines.utils import concordance_index
import joblib
import warnings
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimSun']
rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置参数
# ==========================================

main_title = "三种机器学习模型C-指数"
ml_models_config = {
    'Gradient Boosting Survival': {'file': '4_gbs_model.pkl', 'short': 'GBS'},
    'Random Survival Forest': {'file': '4_rsf_model.pkl', 'short': 'RSF'},
    'Survival SVM': {'file': '4_ssvm_model.pkl', 'short': 'SSVM'}
}

plot_order = ['GBS', 'RSF', 'SSVM']
model_colors = {
    'GBS': '#ff7f0e',
    'RSF': '#1f77b4',
    'SSVM': '#2ca02c'
}

N_BOOTSTRAP = 1000

# ==========================================
# 2. 加载数据与模型
# ==========================================

output_dir = r""

train_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "2_test_data_scaled.csv"))

features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = features_df['Final_Feature_Name'].tolist()

X_train = train_df[final_features]
X_test = test_df[final_features]

time_col = 'DFS_time'
event_col = 'DFS_event'

T_train = train_df[time_col].values
E_train = train_df[event_col].values
T_test = test_df[time_col].values
E_test = test_df[event_col].values

print(f"\n数据加载完成:")
print(f" 训练集样本数: {len(train_df)}")
print(f" 测试集样本数: {len(test_df)}")


# ==========================================
# 3. 定义Bootstrap计算C指数标准误和95%CI的函数
# ==========================================

def calculate_cindex_with_bootstrap(model, X, T, E, n_bootstrap=1000):
    try:
        risk_scores = model.predict(X)
        c_index = concordance_index(T, -risk_scores, E)
    except Exception as e:
        print(f" 计算原始C指数出错: {e}")
        return np.nan, np.nan, np.nan, np.nan

    c_indices = []
    n_samples = len(T)

    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X.iloc[indices]
        T_boot = T[indices]
        E_boot = E[indices]

        try:
            risk_boot = model.predict(X_boot)
            c_boot = concordance_index(T_boot, -risk_boot, E_boot)
            c_indices.append(c_boot)
        except:
            continue

    if len(c_indices) > 0:
        std_error = np.std(c_indices)
        ci_lower = np.percentile(c_indices, 2.5)
        ci_upper = np.percentile(c_indices, 97.5)
    else:
        std_error = 0.0
        ci_lower = c_index
        ci_upper = c_index

    return c_index, std_error, ci_lower, ci_upper


# ==========================================
# 4. 计算所有模型的C指数和标准误
# ==========================================

all_results = []

ordered_model_names = ['Gradient Boosting Survival', 'Random Survival Forest', 'Survival SVM']

for model_name in ordered_model_names:
    config = ml_models_config[model_name]
    print(f"\n正在处理: {model_name} ({config['short']})")

    model_path = os.path.join(output_dir, config['file'])

    if not os.path.exists(model_path):
        print(f" ✗ 模型文件不存在: {model_path}")
        continue

    try:
        model = joblib.load(model_path)
        print(f" ✓ 模型加载成功")
    except Exception as e:
        print(f" ✗ 模型加载失败: {e}")
        continue

    print(f" 正在计算训练集指标 (Bootstrap {N_BOOTSTRAP} 次)...")
    train_c, train_se, train_ci_l, train_ci_u = calculate_cindex_with_bootstrap(model, X_train, T_train, E_train,
                                                                                N_BOOTSTRAP)
    print(f" 训练集 C-index: {train_c:.4f} (95% CI: {train_ci_l:.4f}-{train_ci_u:.4f})")

    print(f" 正在计算测试集指标 (Bootstrap {N_BOOTSTRAP} 次)...")
    test_c, test_se, test_ci_l, test_ci_u = calculate_cindex_with_bootstrap(model, X_test, T_test, E_test, N_BOOTSTRAP)
    print(f" 测试集 C-index: {test_c:.4f} (95% CI: {test_ci_l:.4f}-{test_ci_u:.4f})")

    all_results.append({
        'Model_Full': model_name,
        'Model': config['short'],
        'Train_C_index': train_c,
        'Train_SE': train_se,
        'Train_CI_Lower': train_ci_l,
        'Train_CI_Upper': train_ci_u,
        'Test_C_index': test_c,
        'Test_SE': test_se,
        'Test_CI_Lower': test_ci_l,
        'Test_CI_Upper': test_ci_u,
        'Gap': train_c - test_c
    })

summary_df = pd.DataFrame(all_results)
summary_df['Model'] = pd.Categorical(summary_df['Model'], categories=plot_order, ordered=True)
summary_df = summary_df.sort_values('Model')

display_df = summary_df[['Model', 'Train_C_index', 'Train_CI_Lower', 'Train_CI_Upper', 'Test_C_index', 'Test_CI_Lower',
                         'Test_CI_Upper']].copy()
display_df['Train_Display'] = display_df.apply(
    lambda row: f"{row['Train_C_index']:.4f} ({row['Train_CI_Lower']:.4f}-{row['Train_CI_Upper']:.4f})", axis=1)
display_df['Test_Display'] = display_df.apply(
    lambda row: f"{row['Test_C_index']:.4f} ({row['Test_CI_Lower']:.4f}-{row['Test_CI_Upper']:.4f})", axis=1)

print("\n所有模型C指数汇总:")
print(display_df[['Model', 'Train_Display', 'Test_Display']].to_string(index=False))

summary_path = os.path.join(output_dir, "9_C_Index_Comparison_ML_Only.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n汇总表已保存: {summary_path}")

# ==========================================
# 5. 可视化对比 
# ==========================================

print("\n生成可视化对比...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
plt.subplots_adjust(wspace=0.35, bottom=0.25, top=0.92)

x = np.arange(len(summary_df))
width = 0.35

bar_colors = [model_colors[m] for m in summary_df['Model']]

train_errors = np.array([
    [row['Train_C_index'] - row['Train_CI_Lower'], row['Train_CI_Upper'] - row['Train_C_index']]
    for _, row in summary_df.iterrows()
]).T

bars1 = ax1.bar(x - width / 2, summary_df['Train_C_index'], width, yerr=train_errors, capsize=5, color=bar_colors,
                edgecolor='black', linewidth=1.5)

ax1.set_xlabel('模型', fontsize=22, fontweight='bold', labelpad=15)
ax1.set_ylabel('C-index', fontsize=22, fontweight='bold', labelpad=15)
ax1.set_title('训练集', fontsize=26, fontweight='bold', pad=20)
ax1.set_xticks(x - width / 2)
ax1.set_xticklabels(summary_df['Model'])
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y', linewidth=1)
ax1.set_ylim([0, 1.1])
ax1.tick_params(labelsize=20, width=1.5, length=6)

for idx, bar in enumerate(bars1):
    height = bar.get_height()
    if not np.isnan(height):
        y_pos = height + train_errors[1][idx] + 0.03
        ax1.text(bar.get_x() + bar.get_width() / 2., y_pos, f'{height:.3f}', ha='center', va='bottom', fontsize=20,
                 fontweight='bold')

test_errors = np.array([
    [row['Test_C_index'] - row['Test_CI_Lower'], row['Test_CI_Upper'] - row['Test_C_index']]
    for _, row in summary_df.iterrows()
]).T

bars2 = ax2.bar(x + width / 2, summary_df['Test_C_index'], width, yerr=test_errors, capsize=5, color=bar_colors,
                edgecolor='black', linewidth=1.5)

ax2.set_xlabel('模型', fontsize=22, fontweight='bold', labelpad=15)
ax2.set_ylabel('C-index', fontsize=22, fontweight='bold', labelpad=15)
ax2.set_title('测试集', fontsize=26, fontweight='bold', pad=20)
ax2.set_xticks(x + width / 2)
ax2.set_xticklabels(summary_df['Model'])
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y', linewidth=1)
ax2.set_ylim([0, 1.1])
ax2.tick_params(labelsize=20, width=1.5, length=6)

for idx, bar in enumerate(bars2):
    height = bar.get_height()
    if not np.isnan(height):
        y_pos = height + test_errors[1][idx] + 0.03
        ax2.text(bar.get_x() + bar.get_width() / 2., y_pos, f'{height:.3f}', ha='center', va='bottom', fontsize=20,
                 fontweight='bold')

plt.tight_layout()
plot_path = os.path.join(output_dir, "9_C_Index_Comparison_ML_Only.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"对比图已保存: {plot_path}")
