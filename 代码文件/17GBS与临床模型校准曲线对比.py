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
# 1. 加载数据与模型
# ==========================================

train_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "2_test_data_scaled.csv"))

final_features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = final_features_df['Final_Feature_Name'].tolist()

time_col = 'DFS_time'
event_col = 'DFS_event'

model_names = ['GBS', 'Leibovich', 'UISS', 'KEYNOTE-564']

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

markers = {
    'GBS': 'o',
    'Leibovich': 's',
    'UISS': '^',
    'KEYNOTE-564': 'D'
}

print("\n正在加载模型...")

gbs_model = None
try:
    gbs_model = joblib.load(os.path.join(output_dir, "4_gbs_model.pkl"))
    print("✓ GBS 模型加载成功")
except Exception as e:
    print(f"✗ GBS 模型加载失败: {e}")

print("✓ 临床模型 直接从数据列读取")


# ==========================================
# 2. 预测函数
# ==========================================

def predict_survival_probability(model_name, df, time_point):
    try:
        if model_name == 'GBS':
            if gbs_model is None:
                return None
            risk_scores = gbs_model.predict(df[final_features])
            min_risk = risk_scores.min()
            max_risk = risk_scores.max()
            if max_risk > min_risk:
                normalized_risk = (risk_scores - min_risk) / (max_risk - min_risk)
            else:
                normalized_risk = np.zeros_like(risk_scores)
            pred_survival = 1 - normalized_risk
            return pred_survival
        else:
            if model_name in df.columns:
                risk_scores = df[model_name].values
                risk_map = {1: 0.8, 2: 0.5, 3: 0.2}
                pred_survival = np.array([risk_map.get(r, 0.5) for r in risk_scores])
                return pred_survival
            else:
                print(f"✗ 未找到列 '{model_name}'")
                return None
    except Exception as e:
        print(f"预测 {model_name} 生存概率时出错: {e}")
        return None


# ==========================================
# 3. 绘制校准曲线对比图
# ==========================================

def plot_calibration_comparison(df, features, time_col, event_col, time_point, title, save_path):
    fig, ax = plt.subplots(figsize=(14, 12))

    ax.plot([0, 1], [0, 1], 'k--', linewidth=3, label='理想校准曲线', alpha=0.6)

    for model_name in model_names:
        pred_survival = predict_survival_probability(model_name, df, time_point)

        if pred_survival is None:
            continue

        pred_survival = np.clip(pred_survival, 0.001, 0.999)

        df_temp = df.copy()
        df_temp['pred_survival'] = pred_survival

        if model_name == 'GBS':
            try:
                df_temp['group'] = pd.qcut(df_temp['pred_survival'], q=4, duplicates='drop')
            except Exception as e:
                print(f" {model_name} 分组失败: {e}")
                continue
        else:
            df_temp['group'] = df[model_name].astype(str)

        if len(df_temp['group'].unique()) < 2:
            print(f" {model_name} 分组数量不足")
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

        ax.scatter(group_means, observed_survivals, color=colors[model_name], marker=markers[model_name],
                   s=180 if model_name == 'GBS' else 150, label=model_name, alpha=0.8, edgecolors='black', linewidth=2)

        sorted_idx = np.argsort(group_means.values)
        ax.plot(group_means.values[sorted_idx], observed_survivals.values[sorted_idx], color=colors[model_name],
                linestyle=linestyles[model_name], alpha=0.6, linewidth=2.5 if model_name == 'GBS' else 2.0)

        print(f" {model_name}: {len(group_means)} 个校准点")

    ax.set_xlabel("模型预测生存概率", fontsize=24, fontweight='bold', labelpad=15)
    ax.set_ylabel("实际生存概率", fontsize=24, fontweight='bold', labelpad=15)
    ax.set_title(title, fontsize=26, fontweight='bold', pad=20)
    ax.legend(fontsize=14, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 4. 主程序
# ==========================================

print("\n开始绘制校准曲线对比图...")

time_point_3y = 36
print(f"\n3年 ({time_point_3y}月) 校准曲线:")

plot_calibration_comparison(
    train_df, final_features, time_col, event_col, time_point_3y,
    '训练集（36月）',
    os.path.join(output_dir, f"8_GBS_vs_Clinical_Calibration_Train_3y.png")
)

plot_calibration_comparison(
    test_df, final_features, time_col, event_col, time_point_3y,
    '测试集（36月）',
    os.path.join(output_dir, f"8_GBS_vs_Clinical_Calibration_Test_3y.png")
)

time_point_5y = 60
print(f"\n5年 ({time_point_5y}月) 校准曲线:")

plot_calibration_comparison(
    train_df, final_features, time_col, event_col, time_point_5y,
    '训练集（60月）',
    os.path.join(output_dir, f"8_GBS_vs_Clinical_Calibration_Train_5y.png")
)

plot_calibration_comparison(
    test_df, final_features, time_col, event_col, time_point_5y,
    '测试集（60月）',
    os.path.join(output_dir, f"8_GBS_vs_Clinical_Calibration_Test_5y.png")
)

print("\n校准曲线对比图已保存")

print("\n" + "=" * 60)
print("所有任务完成！")
print("=" * 60)

print("\n校准曲线解读:")
print("- 理想校准曲线 (黑色虚线): 预测概率 = 实际概率")
print("- 点越接近对角线，校准越好")
print("- GBS 使用橙色实线，临床模型使用虚线")
print("- 临床模型使用风险分级：1=低危(0.8), 2=中危(0.5), 3=高危(0.2)")
