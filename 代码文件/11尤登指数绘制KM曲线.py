import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from lifelines import KaplanMeierFitter
import joblib
import warnings

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# ==========================================
# 1. 加载数据与模型
# ==========================================

output_dir = r""

train_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "2_test_data_scaled.csv"))

final_features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = final_features_df['Final_Feature_Name'].tolist()

time_col = 'PFS_time'
event_col = 'PFS_event'

models = {}
model_names = ['COX', 'RSF', 'GBS', 'SSVM']

print("\n正在加载模型...")

for name in model_names:
    try:
        if name == 'COX':
            models[name] = joblib.load(os.path.join(output_dir, "4_cox_model.pkl"))
        elif name == 'RSF':
            models[name] = joblib.load(os.path.join(output_dir, "4_rsf_model.pkl"))
        elif name == 'GBS':
            models[name] = joblib.load(os.path.join(output_dir, "4_gbs_model.pkl"))
        elif name == 'SSVM':
            models[name] = joblib.load(os.path.join(output_dir, "4_ssvm_model.pkl"))
        print(f"✓ {name} 模型加载成功")
    except Exception as e:
        print(f"✗ {name} 模型加载失败: {e}")
        models[name] = None


# ==========================================
# 2. 定义预测函数
# ==========================================

def predict_risk_score(model, model_name, X):
    try:
        if model_name == 'COX':
            risk = model.predict_partial_hazard(X)
            if hasattr(risk, 'values'):
                risk = risk.values
            return risk
        elif model_name in ['RSF', 'GBS', 'SSVM']:
            risk = model.predict(X)
            return risk
        else:
            return None
    except Exception as e:
        print(f"预测 {model_name} 风险分数时出错: {e}")
        return None


def predict_survival_probability(model, model_name, X, time_point):
    try:
        if model_name == 'COX':
            baseline_survival = model.baseline_survival_
            times = baseline_survival.index.values

            if time_point > times[-1]:
                s0 = baseline_survival.iloc[-1].values[0]
            elif time_point < times[0]:
                s0 = 1.0
            else:
                idx = (np.abs(times - time_point)).argmin()
                s0 = baseline_survival.iloc[idx].values[0]

            risk_scores = model.predict_partial_hazard(X).values
            pred_survival = s0 ** risk_scores
            return pred_survival
        elif model_name in ['RSF', 'GBS']:
            risk_scores = model.predict(X)
            min_risk = risk_scores.min()
            max_risk = risk_scores.max()
            if max_risk > min_risk:
                normalized_risk = (risk_scores - min_risk) / (max_risk - min_risk)
            else:
                normalized_risk = np.zeros_like(risk_scores)
            pred_survival = 1 - normalized_risk
            return pred_survival
        elif model_name == 'SSVM':
            risk_scores = model.predict(X)
            min_risk = risk_scores.min()
            max_risk = risk_scores.max()
            if max_risk > min_risk:
                normalized_risk = (risk_scores - min_risk) / (max_risk - min_risk)
            else:
                normalized_risk = np.zeros_like(risk_scores)
            pred_survival = 1 - normalized_risk
            return pred_survival
        else:
            return None
    except Exception as e:
        print(f"预测 {model_name} 生存概率时出错: {e}")
        return None


# ==========================================
# 3. 计算尤登指数最高处阈值
# ==========================================

def find_youden_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_idx]
    best_youden = youden_index[best_idx]
    best_sensitivity = tpr[best_idx]
    best_specificity = 1 - fpr[best_idx]

    return {
        'threshold': best_threshold,
        'youden_index': best_youden,
        'sensitivity': best_sensitivity,
        'specificity': best_specificity,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'best_idx': best_idx
    }


# ==========================================
# 4. 主分析流程
# ==========================================

time_point = 36
youden_results = {}
risk_predictions = {}

for model_name in model_names:
    if models[model_name] is None:
        continue

    print(f"\n{'=' * 50}")
    print(f"处理模型: {model_name}")
    print(f"{'=' * 50}")

    train_risk = predict_risk_score(models[model_name], model_name, train_df[final_features])
    test_risk = predict_risk_score(models[model_name], model_name, test_df[final_features])

    train_surv_prob = predict_survival_probability(models[model_name], model_name, train_df[final_features], time_point)
    test_surv_prob = predict_survival_probability(models[model_name], model_name, test_df[final_features], time_point)

    y_train_binary = np.zeros(len(train_df))
    y_train_binary[(train_df[time_col] <= time_point) & (train_df[event_col] == 1)] = 1

    y_test_binary = np.zeros(len(test_df))
    y_test_binary[(test_df[time_col] <= time_point) & (test_df[event_col] == 1)] = 1

    mask_train = ~((train_df[time_col] <= time_point) & (train_df[event_col] == 0))
    mask_test = ~((test_df[time_col] <= time_point) & (test_df[event_col] == 0))

    train_valid_mask = mask_train
    test_valid_mask = mask_test

    y_train_valid = y_train_binary[train_valid_mask]
    y_test_valid = y_test_binary[test_valid_mask]

    train_surv_prob_valid = train_surv_prob[train_valid_mask]
    test_surv_prob_valid = test_surv_prob[test_valid_mask]

    train_event_prob = 1 - train_surv_prob_valid
    test_event_prob = 1 - test_surv_prob_valid

    youden_info = find_youden_threshold(y_train_valid, train_event_prob)
    youden_results[model_name] = youden_info

    print(f"\n基于训练集的尤登指数分析:")
    print(f" 最佳阈值 (事件发生概率): {youden_info['threshold']:.4f}")
    print(f" 对应的生存概率阈值: {1 - youden_info['threshold']:.4f}")
    print(f" 尤登指数: {youden_info['youden_index']:.4f}")
    print(f" 敏感性: {youden_info['sensitivity']:.4f}")
    print(f" 特异性: {youden_info['specificity']:.4f}")

    y_test_pred = (test_event_prob >= youden_info['threshold']).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test_valid, y_test_pred).ravel()

    test_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    test_accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"\n测试集性能:")
    print(f" 敏感性: {test_sensitivity:.4f}")
    print(f" 特异性: {test_specificity:.4f}")
    print(f" 准确率: {test_accuracy:.4f}")

    surv_threshold = 1 - youden_info['threshold']

    train_risk_group = np.where(train_surv_prob >= surv_threshold, 'Low Risk', 'High Risk')
    test_risk_group = np.where(test_surv_prob >= surv_threshold, 'Low Risk', 'High Risk')

    train_df[f'{model_name}_surv_prob'] = train_surv_prob
    train_df[f'{model_name}_risk_group_youden'] = train_risk_group

    test_df[f'{model_name}_surv_prob'] = test_surv_prob
    test_df[f'{model_name}_risk_group_youden'] = test_risk_group

    risk_predictions[model_name] = {
        'train': {
            'surv_prob': train_surv_prob,
            'risk_group': train_risk_group,
            'y_true': y_train_valid,
            'y_pred': (1 - train_surv_prob_valid >= youden_info['threshold']).astype(int)
        },
        'test': {
            'surv_prob': test_surv_prob,
            'risk_group': test_risk_group,
            'y_true': y_test_valid,
            'y_pred': y_test_pred
        },
        'threshold': youden_info['threshold'],
        'survival_threshold': surv_threshold
    }

# ==========================================
# 5. 绘制ROC曲线并标注最佳阈值
# ==========================================

print("\n绘制ROC曲线...")

fig_roc, axes_roc = plt.subplots(2, 2, figsize=(20, 16))
plt.subplots_adjust(wspace=0.35, hspace=0.4, bottom=0.15, top=0.93)

axes_roc = axes_roc.flatten()

for idx, model_name in enumerate(model_names):
    ax = axes_roc[idx]

    if model_name not in youden_results:
        ax.text(0.5, 0.5, f'{model_name}\n无结果', ha='center', va='center', fontsize=20)
        continue

    youden_info = youden_results[model_name]

    ax.plot(youden_info['fpr'], youden_info['tpr'],
            label=f'ROC (AUC = {auc(youden_info["fpr"], youden_info["tpr"]):.3f})', linewidth=3, color='blue')

    best_idx = youden_info['best_idx']
    ax.scatter(youden_info['fpr'][best_idx], youden_info['tpr'][best_idx], color='red', s=250, marker='o', zorder=5,
               label=f'最佳阈值\n(Youden={youden_info["youden_index"]:.3f})', edgecolors='black', linewidth=2.5)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5)

    ax.annotate(f'({youden_info["fpr"][best_idx]:.2f}, {youden_info["tpr"][best_idx]:.2f})',
                xy=(youden_info['fpr'][best_idx], youden_info['tpr'][best_idx]),
                xytext=(youden_info['fpr'][best_idx] + 0.15, youden_info['tpr'][best_idx] - 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=14, color='red', fontweight='bold')

    ax.set_xlabel('假阳性率 (1-特异性)', fontsize=22, fontweight='bold', labelpad=15)
    ax.set_ylabel('真阳性率 (敏感性)', fontsize=22, fontweight='bold', labelpad=15)
    ax.set_title(f'{model_name} - ROC曲线与尤登指数', fontsize=26, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.tick_params(labelsize=20, width=1.5, length=6)

plt.suptitle(f'4模型ROC曲线对比 (时间点: {time_point}个月)', fontsize=28, fontweight='bold', y=0.98)

plt.tight_layout()
roc_path = os.path.join(output_dir, "6_ROC_Youden_Index.png")
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"ROC曲线已保存: {roc_path}")

# ==========================================
# 6. 绘制KM曲线 (基于尤登指数分组)
# ==========================================

print("\n绘制KM生存曲线...")

fig_km_train, axes_km_train = plt.subplots(2, 2, figsize=(20, 16))
plt.subplots_adjust(wspace=0.35, hspace=0.4, bottom=0.15, top=0.93)

axes_km_train = axes_km_train.flatten()

kmf = KaplanMeierFitter()

for idx, model_name in enumerate(model_names):
    ax = axes_km_train[idx]

    if model_name not in risk_predictions:
        continue

    for group in ['Low Risk', 'High Risk']:
        group_data = train_df[train_df[f'{model_name}_risk_group_youden'] == group]
        if len(group_data) > 0:
            kmf.fit(group_data[time_col], event_observed=group_data[event_col], label=group)
            kmf.plot_survival_function(ax=ax, ci_show=True, lw=3)

    ax.set_title(f'{model_name} - 训练集 (尤登指数)', fontsize=26, fontweight='bold', pad=20)
    ax.set_xlabel("生存时间 (月)", fontsize=22, fontweight='bold', labelpad=15)
    ax.set_ylabel("生存概率", fontsize=22, fontweight='bold', labelpad=15)
    ax.grid(True, alpha=0.3, linewidth=1)
    ax.legend(loc='upper right', fontsize=10)
    ax.tick_params(labelsize=20, width=1.5, length=6)

plt.suptitle('4模型KM生存曲线对比 (基于尤登指数分组)', fontsize=28, fontweight='bold', y=0.98)

plt.tight_layout()
km_train_path = os.path.join(output_dir, "6_KM_Youden_Index_Train.png")
plt.savefig(km_train_path, dpi=300, bbox_inches='tight')
plt.close()

fig_km_test, axes_km_test = plt.subplots(2, 2, figsize=(20, 16))
plt.subplots_adjust(wspace=0.35, hspace=0.4, bottom=0.15, top=0.93)

axes_km_test = axes_km_test.flatten()

for idx, model_name in enumerate(model_names):
    ax = axes_km_test[idx]

    if model_name not in risk_predictions:
        continue

    for group in ['Low Risk', 'High Risk']:
        group_data = test_df[test_df[f'{model_name}_risk_group_youden'] == group]
        if len(group_data) > 0:
            kmf.fit(group_data[time_col], event_observed=group_data[event_col], label=group)
            kmf.plot_survival_function(ax=ax, ci_show=True, lw=3)

    ax.set_title(f'{model_name} - 测试集 (尤登指数)', fontsize=26, fontweight='bold', pad=20)
    ax.set_xlabel("生存时间 (月)", fontsize=22, fontweight='bold', labelpad=15)
    ax.set_ylabel("生存概率", fontsize=22, fontweight='bold', labelpad=15)
    ax.grid(True, alpha=0.3, linewidth=1)
    ax.legend(loc='upper right', fontsize=10)
    ax.tick_params(labelsize=20, width=1.5, length=6)

plt.suptitle('4模型KM生存曲线对比 (基于尤登指数分组)', fontsize=28, fontweight='bold', y=0.98)

plt.tight_layout()
km_test_path = os.path.join(output_dir, "6_KM_Youden_Index_Test.png")
plt.savefig(km_test_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"训练集KM曲线已保存: {km_train_path}")
print(f"测试集KM曲线已保存: {km_test_path}")

# ==========================================
# 7. 保存数据
# ==========================================

print("\n保存预测结果...")

train_output = train_df.copy()
key_columns = [time_col, event_col]

for model_name in model_names:
    if model_name in risk_predictions:
        key_columns.append(f'{model_name}_surv_prob')
        key_columns.append(f'{model_name}_risk_group_youden')

train_output = train_output[key_columns]
train_save_path = os.path.join(output_dir, "6_train_predictions_youden.csv")
train_output.to_csv(train_save_path, index=False)
print(f"训练集预测结果已保存: {train_save_path}")

test_output = test_df.copy()
test_output = test_output[key_columns]
test_save_path = os.path.join(output_dir, "6_test_predictions_youden.csv")
test_output.to_csv(test_save_path, index=False)
print(f"测试集预测结果已保存: {test_save_path}")

# ==========================================
# 8. 生成分层统计表
# ==========================================

print("\n生成分层统计表...")

summary_data = []

for model_name in model_names:
    if model_name not in risk_predictions:
        continue

    pred = risk_predictions[model_name]

    train_low_risk = (train_df[f'{model_name}_risk_group_youden'] == 'Low Risk').sum()
    train_high_risk = (train_df[f'{model_name}_risk_group_youden'] == 'High Risk').sum()

    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    train_low_data = train_df[train_df[f'{model_name}_risk_group_youden'] == 'Low Risk']
    train_high_data = train_df[train_df[f'{model_name}_risk_group_youden'] == 'High Risk']

    if len(train_low_data) > 0:
        kmf_low.fit(train_low_data[time_col], event_observed=train_low_data[event_col])
        train_low_surv_36m = kmf_low.survival_function_at_times(time_point).values[
            0] if time_point in kmf_low.survival_function_.index else kmf_low.survival_function_.iloc[-1, 0]
    else:
        train_low_surv_36m = np.nan

    if len(train_high_data) > 0:
        kmf_high.fit(train_high_data[time_col], event_observed=train_high_data[event_col])
        train_high_surv_36m = kmf_high.survival_function_at_times(time_point).values[
            0] if time_point in kmf_high.survival_function_.index else kmf_high.survival_function_.iloc[-1, 0]
    else:
        train_high_surv_36m = np.nan

    test_low_risk = (test_df[f'{model_name}_risk_group_youden'] == 'Low Risk').sum()
    test_high_risk = (test_df[f'{model_name}_risk_group_youden'] == 'High Risk').sum()

    test_low_data = test_df[test_df[f'{model_name}_risk_group_youden'] == 'Low Risk']
    test_high_data = test_df[test_df[f'{model_name}_risk_group_youden'] == 'High Risk']

    if len(test_low_data) > 0:
        kmf_low.fit(test_low_data[time_col], event_observed=test_low_data[event_col])
        test_low_surv_36m = kmf_low.survival_function_at_times(time_point).values[
            0] if time_point in kmf_low.survival_function_.index else kmf_low.survival_function_.iloc[-1, 0]
    else:
        test_low_surv_36m = np.nan

    if len(test_high_data) > 0:
        kmf_high.fit(test_high_data[time_col], event_observed=test_high_data[event_col])
        test_high_surv_36m = kmf_high.survival_function_at_times(time_point).values[
            0] if time_point in kmf_high.survival_function_.index else kmf_high.survival_function_.iloc[-1, 0]
    else:
        test_high_surv_36m = np.nan

    summary_data.append({
        'Model': model_name,
        'Train_Low_Risk_N': train_low_risk,
        'Train_High_Risk_N': train_high_risk,
        'Train_Low_Risk_Surv_36m': f"{train_low_surv_36m:.3f}",
        'Train_High_Risk_Surv_36m': f"{train_high_surv_36m:.3f}",
        'Test_Low_Risk_N': test_low_risk,
        'Test_High_Risk_N': test_high_risk,
        'Test_Low_Risk_Surv_36m': f"{test_low_surv_36m:.3f}",
        'Test_High_Risk_Surv_36m': f"{test_high_surv_36m:.3f}",
        'Youden_Threshold_Event_Prob': f"{pred['threshold']:.4f}",
        'Youden_Threshold_Surv_Prob': f"{pred['survival_threshold']:.4f}",
        'Youden_Index': f"{youden_results[model_name]['youden_index']:.4f}"
    })

summary_df = pd.DataFrame(summary_data)
summary_path = os.path.join(output_dir, "6_Youden_Index_Summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"分层统计表已保存: {summary_path}")

print("\n分层统计表内容:")
print(summary_df.to_string(index=False))

