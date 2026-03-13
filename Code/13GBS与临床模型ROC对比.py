import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
import warnings
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimSun']
rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置参数
# ==========================================

time_col = 'DFS_time'
event_col = 'DFS_event'

time_points = [36, 60]

N_BOOTSTRAP = 1000

model_list = ['GBS', 'Leibovich', 'UISS', 'KEYNOTE-564']

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
        min_risk = risk_scores.min()
        max_risk = risk_scores.max()
        if max_risk > min_risk:
            norm_risk = (risk_scores - min_risk) / (max_risk - min_risk)
        else:
            norm_risk = np.zeros_like(risk_scores)
        scores = norm_risk
    else:
        if model_name in df.columns:
            scores = df[model_name].values
        else:
            print(f" ✗ 错误: 数据中未找到列 '{model_name}'")

    return scores


# ==========================================
# 4. 计算95%置信区间方法
# ==========================================

def bootstrap_auc_ci(y_true, y_scores, n_bootstrap=1000):
    bootstrapped_scores = []

    valid_indices = np.where(~np.isnan(y_true) & ~np.isnan(y_scores))[0]
    if len(valid_indices) == 0:
        return np.nan, np.nan, np.nan

    y_true_valid = y_true[valid_indices]
    y_scores_valid = y_scores[valid_indices]

    if len(np.unique(y_true_valid)) < 2:
        return np.nan, np.nan, np.nan

    for i in range(n_bootstrap):
        indices = resample(np.arange(len(y_true_valid)), random_state=i)
        if len(np.unique(y_true_valid[indices])) < 2:
            continue
        try:
            fpr_boot, tpr_boot, _ = roc_curve(y_true_valid[indices], y_scores_valid[indices])
            score = auc(fpr_boot, tpr_boot)
            bootstrapped_scores.append(score)
        except:
            continue

    if not bootstrapped_scores:
        return np.nan, np.nan, np.nan

    mean_auc = np.mean(bootstrapped_scores)
    ci_lower = np.percentile(bootstrapped_scores, 2.5)
    ci_upper = np.percentile(bootstrapped_scores, 97.5)

    return mean_auc, ci_lower, ci_upper


# ==========================================
# 5. 计算时间依赖 ROC (含95% CI)
# ==========================================

def calculate_time_roc_with_ci(df, time_point, model_name):
    scores = get_model_scores(df, model_name)
    if scores is None:
        return None, None, np.nan, np.nan, np.nan

    T = df[time_col].values
    E = df[event_col].values

    y_true = np.zeros(len(T))
    y_true[(T <= time_point) & (E == 1)] = 1

    mask = (T > time_point) | ((T <= time_point) & (E == 1))
    if np.sum(mask) == 0:
        print(f" 警告: {model_name} 在 {time_point}月 无有效样本")
        return None, None, np.nan, np.nan, np.nan

    y_true_valid = y_true[mask]
    scores_valid = scores[mask]

    fpr, tpr, thresholds = roc_curve(y_true_valid, scores_valid)
    roc_auc = auc(fpr, tpr)

    auc_mean, ci_lower, ci_upper = bootstrap_auc_ci(
        y_true_valid, scores_valid, N_BOOTSTRAP
    )

    return fpr, tpr, auc_mean, ci_lower, ci_upper


# ==========================================
# 6. 绘制 ROC 对比图
# ==========================================

def plot_comparison_roc_by_timepoint(train_df, test_df, time_point, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    plt.subplots_adjust(wspace=0.35, bottom=0.25, top=0.92)

    colors = {
        'GBS': '#ff7f0e',
        'Leibovich': '#d62728',
        'UISS': '#17becf',
        'KEYNOTE-564': '#9467bd'
    }

    linestyles = {
        'GBS': '-',
        'Leibovich': '--',
        'UISS': '-.',
        'KEYNOTE-564': ':'
    }

    linewidths = {
        'GBS': 3.0,
        'Leibovich': 2.0,
        'UISS': 2.0,
        'KEYNOTE-564': 2.0
    }

    datasets = [
        ('训练集', train_df),
        ('测试集', test_df)
    ]

    for idx, (dataset_name, df) in enumerate(datasets):
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.5)

        aucs = {}

        for model in model_list:
            fpr, tpr, auc_mean, ci_lower, ci_upper = calculate_time_roc_with_ci(df, time_point, model)

            if fpr is not None:
                label = f"{model} (AUC = {auc_mean:.3f}, 95% CI: {ci_lower:.3f}-{ci_upper:.3f})"
                ax.plot(fpr, tpr, color=colors[model], lw=linewidths[model], linestyle=linestyles[model], label=label)
                aucs[model] = (auc_mean, ci_lower, ci_upper)

        ax.set_xlabel('1-特异性', fontsize=22, fontweight='bold', labelpad=15)
        ax.set_ylabel('敏感性', fontsize=22, fontweight='bold', labelpad=15)
        ax.set_title(f'{dataset_name}（{time_point}月）', fontsize=26, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.tick_params(labelsize=20)

        print(f"\n{dataset_name} - {time_point}个月 AUC 值 (含95% CI):")
        for model, (auc_val, ci_lower, ci_upper) in aucs.items():
            print(f" {model}: {auc_val:.4f} ({ci_lower:.4f} - {ci_upper:.4f})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {save_path}")


# ==========================================
# 7. 主程序执行
# ==========================================

for time_point in time_points:
    print(f"\n{'=' * 60}")
    print(f"正在处理 {time_point}个月 ({time_point // 12}年)...")
    print(f"{'=' * 60}")

    save_path = os.path.join(output_dir, f"6_GBS_vs_Clinical_ROC_{time_point}m.png")
    plot_comparison_roc_by_timepoint(train_df, test_df, time_point, save_path)

print("\n" + "=" * 60)
print("GBS 与临床模型 ROC 对比完成！")
print("=" * 60)
