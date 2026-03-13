import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts
from sklearn.metrics import roc_curve
import joblib
import warnings
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimSun']
rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置参数
# ==========================================

time_col = 'DFS_time'
event_col = 'DFS_event'

TIME_POINT_MONTHS = 36

color_high = '#d62728'
color_low = '#1f77b4'

# ==========================================
# 2. 加载数据与 GBS 模型
# ==========================================

print("\n加载数据与模型...")

train_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "2_test_data_scaled.csv"))

features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = features_df['Final_Feature_Name'].tolist()

try:
    gbs_model = joblib.load(os.path.join(output_dir, "4_gbs_model.pkl"))
    print(" ✓ GBS 模型加载成功")
except Exception as e:
    print(f" ✗ GBS 模型加载失败: {e}")
    exit()

# ==========================================
# 3. 计算训练集 ROC 最优阈值
# ==========================================

print(f"\n计算训练集 ROC 最优阈值 (时间点: {TIME_POINT_MONTHS}个月)...")

X_train = train_df[final_features]
T_train = train_df[time_col].values
E_train = train_df[event_col].values

train_risk_scores = gbs_model.predict(X_train)

y_train_binary = np.zeros(len(T_train))
y_train_binary[(T_train <= TIME_POINT_MONTHS) & (E_train == 1)] = 1

valid_mask = (T_train > TIME_POINT_MONTHS) | ((T_train <= TIME_POINT_MONTHS) & (E_train == 1))
y_train_valid = y_train_binary[valid_mask]
train_scores_valid = train_risk_scores[valid_mask]

fpr, tpr, thresholds = roc_curve(y_train_valid, train_scores_valid)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]
max_youden = youden_index[optimal_idx]

print(f" 最优阈值: {optimal_threshold:.4f}")
print(f" 最大尤登指数: {max_youden:.4f}")


# ==========================================
# 4. 定义分组与分析函数
# ==========================================

def analyze_survival(df, X_data, model, threshold, dataset_name):
    print(f"\n--- 分析 {dataset_name} ---")

    risk_scores = model.predict(X_data)
    df['Risk_Group'] = np.where(risk_scores >= threshold, 'High', 'Low')
    df['Risk_Group'] = df['Risk_Group'].astype('category')

    low_risk_df = df[df['Risk_Group'] == 'Low']
    high_risk_df = df[df['Risk_Group'] == 'High']

    print(f" 低危组样本数: {len(low_risk_df)}")
    print(f" 高危组样本数: {len(high_risk_df)}")

    T = df[time_col].values
    E = df[event_col].values
    groups = df['Risk_Group']

    results = multivariate_logrank_test(T, groups, E)
    p_value = results.p_value
    test_statistic = results.test_statistic

    print(f" Log-rank 检验统计量: {test_statistic:.4f}")
    print(f" P值: {p_value:.4e}")

    df_copy = df.copy()
    df_copy['Risk_Code'] = (df_copy['Risk_Group'] == 'High').astype(int)

    cph = CoxPHFitter()
    try:
        cph.fit(df_copy, duration_col=time_col, event_col=event_col, formula="Risk_Code")
        summary = cph.summary
        hr = summary['exp(coef)'].values[0]
        hr_lower = summary['exp(coef) lower 95%'].values[0]
        hr_upper = summary['exp(coef) upper 95%'].values[0]
        print(f" 风险比 (HR): {hr:.3f} (95% CI: {hr_lower:.3f} - {hr_upper:.3f})")
    except:
        hr, hr_lower, hr_upper = np.nan, np.nan, np.nan
        print(" Cox 回归计算失败")

    return low_risk_df, high_risk_df, p_value, hr, hr_lower, hr_upper


# ==========================================
# 5. 执行分析
# ==========================================

X_train = train_df[final_features]
train_low, train_high, p_train, hr_train, hr_l_train, hr_u_train = analyze_survival(
    train_df, X_train, gbs_model, optimal_threshold, "训练集"
)

X_test = test_df[final_features]
test_low, test_high, p_test, hr_test, hr_l_test, hr_u_test = analyze_survival(
    test_df, X_test, gbs_model, optimal_threshold, "测试集"
)


# ==========================================
# 6. 绘图函数
# ==========================================

def plot_km_curve(ax, low_df, high_df, dataset_name, p_val, hr, hr_l, hr_u):
    kmf = KaplanMeierFitter()

    kmf.fit(low_df[time_col], low_df[event_col], label='低风险')
    ax = kmf.plot_survival_function(ax=ax, color=color_low, lw=3.5, ci_show=True)

    kmf.fit(high_df[time_col], high_df[event_col], label='高风险')
    ax = kmf.plot_survival_function(ax=ax, color=color_high, lw=3.5, ci_show=True)

    info_text = ""
    if not np.isnan(hr):
        info_text += f"HR = {hr:.3f} (95% CI: {hr_l:.3f}-{hr_u:.3f})"
    if p_val < 0.001:
        if info_text:
            info_text += "\n"
        info_text += f"P < 0.001"

    ax.text(0.05, 0.15, info_text, transform=ax.transAxes, fontsize=16, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=1.5))

    ax.legend(loc='lower left', fontsize=10)
    ax.set_title(dataset_name, fontsize=26, fontweight='bold', pad=22)
    ax.set_xlabel("生存时间（月）", fontsize=22, fontweight='bold', labelpad=15)
    ax.set_ylabel("无病生存概率", fontsize=22, fontweight='bold', labelpad=15)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=20)

    kmf_low_obj = KaplanMeierFitter().fit(low_df[time_col], low_df[event_col], label='低风险')
    kmf_high_obj = KaplanMeierFitter().fit(high_df[time_col], high_df[event_col], label='高风险')
    add_at_risk_counts(kmf_low_obj, kmf_high_obj, ax=ax, rows_to_show=['At risk'])

    for text_obj in ax.texts:
        text_content = text_obj.get_text()
        pos = text_obj.get_position()
        x_coord = pos[0]
        y_coord = pos[1]

        if y_coord < 0.1:
            is_table_text = False
            if 'At risk' in text_content or (text_content.replace('.', '').isdigit()):
                is_table_text = True

            if is_table_text:
                text_obj.set_fontsize(22)
                text_obj.set_position((x_coord, y_coord - 1))
            else:
                text_obj.set_fontsize(16)

    return ax


# ==========================================
# 7. 生成可视化
# ==========================================

print("\n生成生存曲线图...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
plt.subplots_adjust(wspace=0.35, bottom=0.35, top=0.92)

plot_km_curve(ax1, train_low, train_high, "训练集", p_train, hr_train, hr_l_train, hr_u_train)
plot_km_curve(ax2, test_low, test_high, "测试集", p_test, hr_test, hr_l_test, hr_u_test)

plot_path = os.path.join(output_dir, "12_GBS_Kaplan_Meier_Curves.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"生存曲线图已保存: {plot_path}")

# ==========================================
# 8. 生成详细报告
# ==========================================

report_path = os.path.join(output_dir, "12_GBS_KM_Analysis_Report.txt")

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("GBS 模型 Kaplan-Meier 生存曲线与 Log-rank 检验报告\n")
    f.write("=" * 80 + "\n\n")

    f.write("分析概述:\n")
    f.write(f"- 最优阈值: {optimal_threshold:.4f} (基于训练集 {TIME_POINT_MONTHS}个月 ROC 尤登指数)\n")
    f.write(f"- 终点事件: 无病生存期\n\n")

    f.write("训练集 结果:\n")
    f.write(f" P值: {p_train:.4e}\n")
    if not np.isnan(hr_train):
        f.write(f" 风险比 (HR): {hr_train:.3f} (95% CI: {hr_l_train:.3f} - {hr_u_train:.3f})\n")

    f.write("测试集结果 (验证集):\n")
    f.write(f" P值: {p_test:.4e}\n")
    if not np.isnan(hr_test):
        f.write(f" 风险比 (HR): {hr_test:.3f} (95% CI: {hr_l_test:.3f} - {hr_u_test:.3f})\n")

print(f" 详细报告已保存: {report_path}")

