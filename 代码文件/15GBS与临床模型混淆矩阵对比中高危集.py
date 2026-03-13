import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimSun']
rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置参数
# ==========================================

TIME_POINT_MONTHS = 36

model_list = ['GBS', 'Leibovich', 'UISS', 'KEYNOTE-564']

colors = {
    'GBS': '#ff7f0e',
    'Leibovich': '#d62728',
    'UISS': '#17becf',
    'KEYNOTE-564': '#9467bd'
}

# ==========================================
# 2. 加载数据与模型
# ==========================================

train_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "2_test_data_scaled.csv"))

time_col = 'DFS_time'
event_col = 'DFS_event'

print(f"\n数据加载完成:")
print(f" 训练集样本数: {len(train_df)}")
print(f" 测试集样本数: {len(test_df)}")
print(f" 时间列: {time_col}")
print(f" 事件列: {event_col}")
print(f" 评估时间点: {TIME_POINT_MONTHS}个月 ({TIME_POINT_MONTHS / 12}年)")

features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = features_df['Final_Feature_Name'].tolist()

X_train = train_df[final_features]
X_test = test_df[final_features]

T_train = train_df[time_col].values
E_train = train_df[event_col].values
T_test = test_df[time_col].values
E_test = test_df[event_col].values

gbs_model = None
try:
    gbs_model = joblib.load(os.path.join(output_dir, "4_gbs_model.pkl"))
    print(" ✓ GBS 模型加载成功")
except Exception as e:
    print(f" ✗ GBS 模型加载失败: {e}")


# ==========================================
# 3. 定义时间依赖的二分类标签
# ==========================================

def create_time_dependent_labels(T, E, time_point):
    y_true = np.zeros(len(T))
    y_true[(T <= time_point) & (E == 1)] = 1
    return y_true


# ==========================================
# 4. 预测风险分数
# ==========================================

def get_risk_scores(model_name, df, X):
    if model_name == 'GBS':
        if gbs_model is None:
            return np.zeros(len(X))
        return gbs_model.predict(X)
    else:
        if model_name in df.columns:
            return df[model_name].values
        else:
            print(f" ✗ 错误: 数据中未找到列 '{model_name}'")
            return np.zeros(len(df))


# ==========================================
# 5. 计算ROC曲线和尤登指数
# ==========================================

def calculate_roc_and_youden(model_name, df, X, y_true, T, E, time_point):
    scores = get_risk_scores(model_name, df, X)

    valid_mask = (T > time_point) | ((T <= time_point) & (E == 1))
    y_true_valid = y_true[valid_mask]
    scores_valid = scores[valid_mask]

    if len(np.unique(y_true_valid)) < 2:
        print(f" 警告: {model_name} 有效样本中只有一个类别，无法计算ROC，返回默认值")
        return None, None, None, None, None, None

    fpr, tpr, thresholds = roc_curve(y_true_valid, scores_valid)
    youden_index = tpr - fpr
    max_index = np.argmax(youden_index)
    optimal_threshold = thresholds[max_index]
    max_youden = youden_index[max_index]

    return fpr, tpr, thresholds, youden_index, optimal_threshold, max_youden


# ==========================================
# 6. 计算混淆矩阵
# ==========================================

def calculate_confusion_matrix_with_threshold(model_name, df, X, y_true, T, E, threshold, time_point):
    scores = get_risk_scores(model_name, df, X)

    if model_name in ['Leibovich', 'UISS', 'KEYNOTE-564']:
        y_pred = (scores >= 1.5).astype(int)
    else:
        y_pred = (scores >= threshold).astype(int)

    valid_mask = (T > time_point) | ((T <= time_point) & (E == 1))
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    cm = confusion_matrix(y_true_valid, y_pred_valid)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'threshold': threshold,
        'n_valid': len(y_true_valid)
    }


# ==========================================
# 7. 格式化混淆矩阵标签
# ==========================================

def format_cm_label(value, total):
    percentage = value / total * 100
    return f"{percentage:.1f}%\n(n={value})"


# ==========================================
# 8. 主程序
# ==========================================

y_train = create_time_dependent_labels(T_train, E_train, TIME_POINT_MONTHS)
y_test = create_time_dependent_labels(T_test, E_test, TIME_POINT_MONTHS)

print(f"\n二分类标签创建完成:")
print(f" 训练集事件发生率: {np.mean(y_train):.2%}")
print(f" 测试集事件发生率: {np.mean(y_test):.2%}")

all_results = {}

print(f"\n开始计算 {TIME_POINT_MONTHS}个月时间点的ROC曲线和尤登指数...")

for model_name in model_list:
    print(f" 正在计算 {model_name} 的ROC曲线和尤登指数...")

    fpr_train, tpr_train, thresholds_train, youden_train, optimal_threshold_train, max_youden_train = calculate_roc_and_youden(
        model_name, train_df, X_train, y_train, T_train, E_train, TIME_POINT_MONTHS
    )

    if optimal_threshold_train is None:
        print(f" ✗ {model_name} 无法计算有效阈值，跳过")
        continue

    print(f" ✓ {model_name} 完成")
    print(f" 最佳阈值: {optimal_threshold_train:.4f}")
    print(f" 尤登指数: {max_youden_train:.4f}")

    metrics_train = calculate_confusion_matrix_with_threshold(
        model_name, train_df, X_train, y_train, T_train, E_train, optimal_threshold_train, TIME_POINT_MONTHS
    )

    metrics_test = calculate_confusion_matrix_with_threshold(
        model_name, test_df, X_test, y_test, T_test, E_test, optimal_threshold_train, TIME_POINT_MONTHS
    )

    all_results[model_name] = {
        'train': metrics_train,
        'test': metrics_test,
        'optimal_threshold': optimal_threshold_train,
        'youden_index': max_youden_train,
        'fpr_train': fpr_train,
        'tpr_train': tpr_train
    }

# ==========================================
# 9. 可视化ROC曲线
# ==========================================

print("\n生成ROC曲线和尤登指数可视化...")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.35, bottom=0.2, top=0.93)

axes_flat = axes.flatten()

for i, model_name in enumerate(model_list):
    if model_name not in all_results:
        continue

    ax = axes_flat[i]
    res = all_results[model_name]

    ax.plot(res['fpr_train'], res['tpr_train'], color=colors[model_name], lw=2, label='ROC曲线')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='随机猜测')

    idx = np.argmax(res['tpr_train'] - res['fpr_train'])
    ax.plot(res['fpr_train'][idx], res['tpr_train'][idx], 'ro', markersize=8,
            label=f'最佳阈值 ({res["optimal_threshold"]:.3f})')

    ax.set_xlabel('假阳性率', fontsize=22, fontweight='bold', labelpad=15)
    ax.set_ylabel('真阳性率', fontsize=22, fontweight='bold', labelpad=15)
    ax.set_title(f'{model_name}\n尤登指数: {res["youden_index"]:.3f}', fontsize=26, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.tick_params(labelsize=20)

for i in range(len(model_list), len(axes_flat)):
    axes_flat[i].axis('off')

plot_path = os.path.join(output_dir, "11_GBS_vs_Clinical_ROC_Youden_Exclude_Censored.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"ROC曲线和尤登指数图已保存: {plot_path}")

# ==========================================
# 10. 可视化混淆矩阵
# ==========================================

print("\n生成混淆矩阵可视化...")

fig, axes = plt.subplots(4, 2, figsize=(18, 24))
plt.subplots_adjust(hspace=0.5, wspace=0.35, bottom=0.15, top=0.94)

for i, model_name in enumerate(model_list):
    if model_name not in all_results:
        continue

    cm_train = all_results[model_name]['train']['confusion_matrix']
    ax_train = axes[i, 0]

    labels_train = []
    for row in range(cm_train.shape[0]):
        row_labels = []
        for col in range(cm_train.shape[1]):
            value = cm_train[row, col]
            total = np.sum(cm_train[row, :])
            label = format_cm_label(value, total)
            row_labels.append(label)
        labels_train.append(row_labels)

    sns.heatmap(cm_train, annot=labels_train, fmt='', cmap='Blues', xticklabels=['低风险', '高风险'],
                yticklabels=['未复发', '复发'], ax=ax_train, cbar=False, annot_kws={'size': 16})

    ax_train.set_xlabel('预测', fontsize=22, fontweight='bold', labelpad=15)
    ax_train.set_ylabel('实际', fontsize=22, fontweight='bold', labelpad=15)
    ax_train.set_title(f'训练集（{model_name}）', fontsize=26, fontweight='bold', pad=20)
    ax_train.tick_params(labelsize=20)

    cm_test = all_results[model_name]['test']['confusion_matrix']
    ax_test = axes[i, 1]

    labels_test = []
    for row in range(cm_test.shape[0]):
        row_labels = []
        for col in range(cm_test.shape[1]):
            value = cm_test[row, col]
            total = np.sum(cm_test[row, :])
            label = format_cm_label(value, total)
            row_labels.append(label)
        labels_test.append(row_labels)

    sns.heatmap(cm_test, annot=labels_test, fmt='', cmap='Blues', xticklabels=['低风险', '高风险'],
                yticklabels=['未复发', '复发'], ax=ax_test, cbar=False, annot_kws={'size': 16})

    ax_test.set_xlabel('预测', fontsize=22, fontweight='bold', labelpad=15)
    ax_test.set_ylabel('实际', fontsize=22, fontweight='bold', labelpad=15)
    ax_test.set_title(f'测试集（{model_name}）', fontsize=26, fontweight='bold', pad=20)
    ax_test.tick_params(labelsize=20)

confusion_matrix_path = os.path.join(output_dir, "11_GBS_vs_Clinical_Confusion_Matrix_Exclude_Censored.png")
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"混淆矩阵图已保存: {confusion_matrix_path}")

