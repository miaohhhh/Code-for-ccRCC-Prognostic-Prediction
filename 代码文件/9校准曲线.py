import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
import joblib
import warnings

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置区域
# ==========================================

TITLES = {
    'train_3y': '训练集（36月）',
    'test_3y': '测试集（36月）',
    'train_5y': '训练集（60月）',
    'test_5y': '测试集（60月）'
}

AXIS_LABELS = {
    'xlabel_3y': '模型预测生存概率',
    'ylabel_3y': '实际生存概率',
    'xlabel_5y': '模型预测生存概率',
    'ylabel_5y': '实际生存概率'
}

# ==========================================
# 2. 加载数据与模型
# ==========================================

output_dir = r""

train_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "2_test_data_scaled.csv"))

final_features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = final_features_df['Final_Feature_Name'].tolist()

time_col = 'DFS_time'
event_col = 'DFS_event'

models = {}
model_names = ['GBS', 'RSF', 'SSVM']

print("\n正在加载模型...")

for name in model_names:
    try:
        if name == 'GBS':
            models[name] = joblib.load(os.path.join(output_dir, "4_gbs_model.pkl"))
        elif name == 'RSF':
            models[name] = joblib.load(os.path.join(output_dir, "4_rsf_model.pkl"))
        elif name == 'SSVM':
            models[name] = joblib.load(os.path.join(output_dir, "4_ssvm_model.pkl"))
        print(f"✓ {name} 模型加载成功")
    except Exception as e:
        print(f"✗ {name} 模型加载失败: {e}")
        models[name] = None


# ==========================================
# 3. 改进的预测函数
# ==========================================

def predict_survival_probability(model, model_name, X, time_point):
    try:
        if model_name in ['RSF', 'GBS']:
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
# 4. 绘制校准曲线对比图
# ==========================================

def plot_calibration_comparison_fixed(models, model_names, df, features, time_col, event_col, time_point, title, xlabel,
                                      ylabel, save_path):
    fig, ax = plt.subplots(figsize=(14, 12))

    colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
    markers = ['s', 'o', '^']

    ax.plot([0, 1], [0, 1], 'k--', linewidth=3, label='理想校准曲线', alpha=0.6)

    for idx, model_name in enumerate(model_names):
        if models[model_name] is None:
            continue

        pred_survival = predict_survival_probability(
            models[model_name], model_name, df[features], time_point
        )

        if pred_survival is None:
            continue

        pred_survival = np.clip(pred_survival, 0.001, 0.999)

        df_temp = df.copy()
        df_temp['pred_survival'] = pred_survival

        try:
            df_temp['group'] = pd.qcut(df_temp['pred_survival'], q=4, duplicates='drop')
        except Exception as e:
            continue

        if len(df_temp['group'].unique()) < 2:
            continue

        group_means = df_temp.groupby('group')['pred_survival'].mean()

        observed_survivals = []
        kmf = KaplanMeierFitter()

        for name, group in df_temp.groupby('group'):
            kmf.fit(group[time_col], event_observed=group[event_col])

            if time_point > kmf.survival_function_.index[-1]:
                obs = kmf.survival_function_.iloc[-1, 0]
            elif time_point < kmf.survival_function_.index[0]:
                obs = 1.0
            else:
                obs = np.interp(time_point, kmf.survival_function_.index, kmf.survival_function_.iloc[:, 0])

            observed_survivals.append(obs)

        observed_survivals = pd.Series(observed_survivals, index=group_means.index)

        ax.scatter(group_means, observed_survivals, color=colors[idx], marker=markers[idx], s=180, label=model_name,
                   alpha=0.8, edgecolors='black', linewidth=2)

        sorted_idx = np.argsort(group_means.values)
        ax.plot(group_means.values[sorted_idx], observed_survivals.values[sorted_idx], color=colors[idx], alpha=0.5,
                linewidth=2.5)

    ax.set_xlabel(xlabel, fontsize=24, fontweight='bold', labelpad=15)
    ax.set_ylabel(ylabel, fontsize=24, fontweight='bold', labelpad=15)
    ax.set_title(title, fontsize=28, fontweight='bold', pad=20)
    ax.legend(fontsize=14, loc='lower right')
    ax.grid(True, alpha=0.3, linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=22, width=1.5, length=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 5. 主程序
# ==========================================

print("\n开始绘制校准曲线对比图...")

time_point_3y = 36
plot_calibration_comparison_fixed(
    models, model_names, train_df, final_features, time_col, event_col, time_point_3y,
    TITLES['train_3y'], AXIS_LABELS['xlabel_3y'], AXIS_LABELS['ylabel_3y'],
    os.path.join(output_dir, f"5_Calibration_Comparison_Train_3y.png")
)

plot_calibration_comparison_fixed(
    models, model_names, test_df, final_features, time_col, event_col, time_point_3y,
    TITLES['test_3y'], AXIS_LABELS['xlabel_3y'], AXIS_LABELS['ylabel_3y'],
    os.path.join(output_dir, f"5_Calibration_Comparison_Test_3y.png")
)

time_point_5y = 60
plot_calibration_comparison_fixed(
    models, model_names, train_df, final_features, time_col, event_col, time_point_5y,
    TITLES['train_5y'], AXIS_LABELS['xlabel_5y'], AXIS_LABELS['ylabel_5y'],
    os.path.join(output_dir, f"5_Calibration_Comparison_Train_5y.png")
)

plot_calibration_comparison_fixed(
    models, model_names, test_df, final_features, time_col, event_col, time_point_5y,
    TITLES['test_5y'], AXIS_LABELS['xlabel_5y'], AXIS_LABELS['ylabel_5y'],
    os.path.join(output_dir, f"5_Calibration_Comparison_Test_5y.png")
)

print("校准曲线对比图已保存")

print("\n" + "=" * 60)
print("所有任务完成！")
print("=" * 60)
