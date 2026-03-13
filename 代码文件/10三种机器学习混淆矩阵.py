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

ml_models = ['Gradient Boosting Survival', 'Random Survival Forest', 'Survival SVM']

model_names_short = {
    'Gradient Boosting Survival': 'GBS',
    'Random Survival Forest': 'RSF',
    'Survival SVM': 'SSVM'
}

ROC_SUBTITLE_TEMPLATE = '{model_short}\n尤登指数: {youden}'
ROC_XLABEL = '假阳性率'
ROC_YLABEL = '真阳性率'
CM_XLABEL = '预测'
CM_YLABEL = '实际'

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

# ==========================================
# 3. 加载机器学习模型
# ==========================================

print("\n加载机器学习模型...")

models = {}
model_files = {
    'Gradient Boosting Survival': '4_gbs_model.pkl',
    'Random Survival Forest': '4_rsf_model.pkl',
    'Survival SVM': '4_ssvm_model.pkl'
}

for model_name, file_name in model_files.items():
    try:
        models[model_name] = joblib.load(os.path.join(output_dir, file_name))
        print(f" ✓ {model_name} 模型加载成功")
    except Exception as e:
        print(f" ✗ {model_name} 模型加载失败: {e}")
        models[model_name] = None


# ==========================================
# 4. 定义时间依赖的二分类标签
# ==========================================

def create_time_dependent_labels(T, E, time_point):
    y_true = np.zeros(len(T))
    y_true[(T <= time_point) & (E == 1)] = 1
    return y_true


# ==========================================
# 5. 预测风险分数
# ==========================================

def predict_risk_scores(model, X):
    if model is None:
        return np.zeros(len(X))

    risk_scores = model.predict(X)

    try:
        survival_probs = model.predict_survival_function(X, times=[TIME_POINT_MONTHS / 12]).iloc[:, 0]
        risk_scores = 1 - survival_probs
    except:
        pass

    return risk_scores


# ==========================================
# 6. 计算ROC曲线和尤登指数
# ==========================================

def calculate_roc_and_youden(model, X, y_true, T, E, time_point):
    risk_scores = predict_risk_scores(model, X)

    valid_mask = (T > time_point) | ((T <= time_point) & (E == 1))
    y_true_valid = y_true[valid_mask]
    risk_scores_valid = risk_scores[valid_mask]

    if len(np.unique(y_true_valid)) < 2:
        return None, None, None, None, None, None

    fpr, tpr, thresholds = roc_curve(y_true_valid, risk_scores_valid)
    youden_index = tpr - fpr
    max_index = np.argmax(youden_index)
    optimal_threshold = thresholds[max_index]
    max_youden = youden_index[max_index]

    return fpr, tpr, thresholds, youden_index, optimal_threshold, max_youden


# ==========================================
# 7. 计算混淆矩阵
# ==========================================

def calculate_confusion_matrix_with_threshold(model, X, y_true, T, E, threshold, time_point):
    risk_scores = predict_risk_scores(model, X)
    y_pred = (risk_scores >= threshold).astype(int)

    valid_mask = (T > time_point) | ((T <= time_point) & (E == 1))
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    cm = confusion_matrix(y_true_valid, y_pred_valid)
    report = classification_report(y_true_valid, y_pred_valid, output_dict=True, zero_division=0)

    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'threshold': threshold
    }


# ==========================================
# 8. 格式化混淆矩阵标签
# ==========================================

def format_cm_label(value, total):
    percentage = value / total * 100
    return f"{percentage:.1f}%\n(n={value})"


# ==========================================
# 9. 获取ROC子图标题
# ==========================================

def get_roc_subtitle(model_short, youden):
    try:
        return ROC_SUBTITLE_TEMPLATE.replace('{model_short}', model_short).replace('{youden}', f'{youden:.3f}')
    except:
        return model_short + '\n尤登指数: ' + f'{youden:.3f}'


# ==========================================
# 10. 主程序
# ==========================================

y_train = create_time_dependent_labels(T_train, E_train, TIME_POINT_MONTHS)
y_test = create_time_dependent_labels(T_test, E_test, TIME_POINT_MONTHS)

print(f"\n二分类标签创建完成:")
print(f" 训练集事件发生率: {np.mean(y_train):.2%}")
print(f" 测试集事件发生率: {np.mean(y_test):.2%}")

all_results = {}

print(f"\n开始计算 {TIME_POINT_MONTHS}个月时间点的ROC曲线和尤登指数...")

for model_name in ml_models:
    if models[model_name] is None:
        print(f" ✗ 跳过 {model_name} (模型未加载)")
        continue

    print(f" 正在计算 {model_name} 的ROC曲线和尤登指数...")

    fpr_train, tpr_train, thresholds_train, youden_train, optimal_threshold_train, max_youden_train = calculate_roc_and_youden(
        models[model_name], X_train, y_train, T_train, E_train, TIME_POINT_MONTHS
    )

    if optimal_threshold_train is None:
        print(f" ✗ {model_name} 无法计算有效阈值，跳过")
        continue

    print(f" ✓ {model_name} 完成")
    print(f" 最佳阈值: {optimal_threshold_train:.4f}")
    print(f" 尤登指数: {max_youden_train:.4f}")

    metrics_train = calculate_confusion_matrix_with_threshold(
        models[model_name], X_train, y_train, T_train, E_train, optimal_threshold_train, TIME_POINT_MONTHS
    )

    metrics_test = calculate_confusion_matrix_with_threshold(
        models[model_name], X_test, y_test, T_test, E_test, optimal_threshold_train, TIME_POINT_MONTHS
    )

    all_results[model_name] = {
        'train': metrics_train,
        'test': metrics_test,
        'optimal_threshold': optimal_threshold_train,
        'youren_index': max_youden_train
    }

# ==========================================
# 11. 可视化ROC曲线和尤登指数
# ==========================================

print("\n生成ROC曲线和尤登指数可视化...")

fig, axes = plt.subplots(1, 3, figsize=(22, 8))
plt.subplots_adjust(wspace=0.35, bottom=0.2, top=0.88)

for i, model_name in enumerate(ml_models):
    if model_name not in all_results:
        continue

    model_short = model_names_short.get(model_name, model_name)

    fpr_train, tpr_train, thresholds_train, youden_train, optimal_threshold_train, max_youden_train = calculate_roc_and_youden(
        models[model_name], X_train, y_train, T_train, E_train, TIME_POINT_MONTHS
    )

    if fpr_train is None:
        continue

    ax = axes[i]

    ax.plot(fpr_train, tpr_train, color='blue', lw=3, label='ROC曲线')
    ax.plot([0, 1], [0, 1], color='gray', lw=2.5, linestyle='--', label='随机猜测')

    idx = np.argmax(youden_train)
    ax.plot(fpr_train[idx], tpr_train[idx], 'ro', markersize=12, label=f'最佳阈值 ({optimal_threshold_train:.3f})')

    ax.set_xlabel(ROC_XLABEL, fontsize=22, fontweight='bold', labelpad=15)
    ax.set_ylabel(ROC_YLABEL, fontsize=22, fontweight='bold', labelpad=15)
    ax.set_title(get_roc_subtitle(model_short, max_youden_train), fontsize=26, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.tick_params(labelsize=20, width=1.5, length=6)

plot_path = os.path.join(output_dir, "9_ROC_Youden_Index_Analysis.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"ROC曲线和尤登指数图已保存: {plot_path}")

# ==========================================
# 12. 可视化混淆矩阵
# ==========================================

print("\n生成混淆矩阵可视化...")

fig, axes = plt.subplots(3, 2, figsize=(18, 20))
plt.subplots_adjust(hspace=0.45, wspace=0.35, bottom=0.12, top=0.94)

for i, model_name in enumerate(ml_models):
    if model_name not in all_results:
        continue

    model_short = model_names_short.get(model_name, model_name)

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

    ax_train.set_xlabel(CM_XLABEL, fontsize=22, fontweight='bold', labelpad=15)
    ax_train.set_ylabel(CM_YLABEL, fontsize=22, fontweight='bold', labelpad=15)
    ax_train.set_title('训练集（' + model_short + '）', fontsize=26, fontweight='bold', pad=20)
    ax_train.tick_params(labelsize=20, width=1.5, length=6)

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

    ax_test.set_xlabel(CM_XLABEL, fontsize=22, fontweight='bold', labelpad=15)
    ax_test.set_ylabel(CM_YLABEL, fontsize=22, fontweight='bold', labelpad=15)
    ax_test.set_title('测试集（' + model_short + '）', fontsize=26, fontweight='bold', pad=20)
    ax_test.tick_params(labelsize=20, width=1.5, length=6)

confusion_matrix_path = os.path.join(output_dir, "9_Confusion_Matrix_Youden_Index.png")
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"混淆矩阵图已保存: {confusion_matrix_path}")
