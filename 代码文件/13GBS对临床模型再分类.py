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

SUBGROUP_ANALYSIS_CONFIG = {
    'Leibovich': [1, 2],
    'UISS': [1, 2],
    'KEYNOTE-564': [1, 2]
}

color_high = '#d62728'
color_low = '#1f77b4'

TIME_NODES = [0, 20, 40, 60, 80, 100]

# ==========================================
# 2. 加载数据与模型
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

datasets = {'Training': train_df, 'Test': test_df}

# ==========================================
# 3. 计算全局最优阈值 (基于训练集)
# ==========================================

X_train = train_df[final_features]
T_train = train_df[time_col].values
E_train = train_df[event_col].values

train_risk_scores = gbs_model.predict(X_train)

y_train_binary = np.zeros(len(T_train))
y_train_binary[(T_train <= TIME_POINT_MONTHS) & (E_train == 1)] = 1

valid_mask = (T_train > TIME_POINT_MONTHS) | ((T_train <= TIME_POINT_MONTHS) & (E_train == 1))
y_train_valid = y_train_binary[valid_mask]
train_scores_valid = train_risk_scores[valid_mask]

if len(np.unique(y_train_valid)) < 2:
    print(" 训练集有效样本中无事件发生，无法计算阈值，使用中位数代替。")
    optimal_threshold = np.median(train_risk_scores)
else:
    fpr, tpr, thresholds = roc_curve(y_train_valid, train_scores_valid)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]

print(f" 全局最优阈值: {optimal_threshold:.4f}")


# ==========================================
# 4. 亚组分析函数
# ==========================================

def analyze_subgroup_survival(df, X_data, model, threshold, group_col, group_vals, dataset_name):
    risk_scores = model.predict(X_data)
    df['GBS_Risk'] = np.where(risk_scores >= threshold, 'High', 'Low')

    sub_df = df[df[group_col].isin(group_vals)].copy()

    if len(sub_df) == 0:
        return None

    if len(sub_df['GBS_Risk'].unique()) < 2:
        score_str = '+'.join(map(str, group_vals))
        return {
            'Dataset': dataset_name,
            'Model': group_col,
            'Scores': score_str,
            'N': len(sub_df),
            'High_N': 0,
            'Low_N': 0,
            'P_value': np.nan,
            'HR': np.nan,
            'HR_lower': np.nan,
            'HR_upper': np.nan,
            'Note': 'Only one risk group present'
        }

    T = sub_df[time_col].values
    E = sub_df[event_col].values
    groups = sub_df['GBS_Risk']

    results = multivariate_logrank_test(T, groups, E)
    p_value = results.p_value

    sub_df_copy = sub_df.copy()
    sub_df_copy['Risk_Code'] = (sub_df_copy['GBS_Risk'] == 'High').astype(int)

    hr, hr_lower, hr_upper = np.nan, np.nan, np.nan

    try:
        cph = CoxPHFitter()
        cph.fit(sub_df_copy, duration_col=time_col, event_col=event_col, formula="Risk_Code")
        summary = cph.summary
        hr = summary['exp(coef)'].values[0]
        hr_lower = summary['exp(coef) lower 95%'].values[0]
        hr_upper = summary['exp(coef) upper 95%'].values[0]
    except Exception as e:
        pass

    time_node_counts = {}
    for time_node in TIME_NODES:
        mask = T >= time_node
        at_risk_df = sub_df[mask].copy()
        low_risk_count = len(at_risk_df[at_risk_df['GBS_Risk'] == 'Low'])
        high_risk_count = len(at_risk_df[at_risk_df['GBS_Risk'] == 'High'])
        time_node_counts[time_node] = {
            'Low_Risk': low_risk_count,
            'High_Risk': high_risk_count
        }

    score_str = '+'.join(map(str, group_vals))

    return {
        'Dataset': dataset_name,
        'Model': group_col,
        'Scores': score_str,
        'N': len(sub_df),
        'High_N': len(sub_df[sub_df['GBS_Risk'] == 'High']),
        'Low_N': len(sub_df[sub_df['GBS_Risk'] == 'Low']),
        'P_value': p_value,
        'HR': hr,
        'HR_lower': hr_lower,
        'HR_upper': hr_upper,
        'Note': 'OK',
        'Time_Node_Counts': time_node_counts
    }


# ==========================================
# 5. 执行循环分析
# ==========================================

all_results = []

train_df_X = train_df[final_features]
test_df_X = test_df[final_features]

for dataset_name, df in datasets.items():
    print(f"\n--- 正在分析 {dataset_name} Set ---")
    X_data = train_df_X if dataset_name == 'Training' else test_df_X

    for model_col, scores in SUBGROUP_ANALYSIS_CONFIG.items():
        if model_col not in df.columns:
            print(f" ⚠️ 警告: 数据中未找到列 '{model_col}'，跳过该模型分析。")
            continue

        print(f" 分析 {model_col} = {'+'.join(map(str, scores))} (合并组) ...")

        res = analyze_subgroup_survival(
            df, X_data, gbs_model, optimal_threshold, model_col, scores, dataset_name
        )

        if res:
            all_results.append(res)

# ==========================================
# 6. 汇总报告保存
# ==========================================

results_df = pd.DataFrame(all_results)


def format_p_val(p):
    if pd.isna(p):
        return "N/A"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


results_df['P_Value_Str'] = results_df['P_value'].apply(format_p_val)

results_df['HR_Str'] = results_df.apply(
    lambda row: f"{row['HR']:.2f} (95% CI: {row['HR_lower']:.2f}-{row['HR_upper']:.2f})" if pd.notna(
        row['HR']) else "N/A", axis=1
)

time_node_tables = []

for model_col in SUBGROUP_ANALYSIS_CONFIG.keys():
    if model_col not in train_df.columns:
        continue

    for dataset_name in ['Training', 'Test']:
        res_row = results_df[
            (results_df['Dataset'] == dataset_name) &
            (results_df['Model'] == model_col)
            ]

        if not res_row.empty:
            res = res_row.iloc[0]
            time_node_counts = res['Time_Node_Counts']

            table_data = []
            for time_node in TIME_NODES:
                if time_node in time_node_counts:
                    counts = time_node_counts[time_node]
                    table_data.append([
                        time_node,
                        counts['Low_Risk'],
                        counts['High_Risk']
                    ])
                else:
                    table_data.append([time_node, 0, 0])

            headers = ['Time (Months)', 'Low Risk', 'High Risk']
            table_df = pd.DataFrame(table_data, columns=headers)
            table_df['Model'] = model_col
            table_df['Dataset'] = dataset_name
            time_node_tables.append(table_df)

if time_node_tables:
    final_table = pd.concat(time_node_tables, ignore_index=True)
    time_node_table_path = os.path.join(output_dir, "13_Time_Node_Counts_Table.csv")
    final_table.to_csv(time_node_table_path, index=False)

    print("\n" + "=" * 60)
    print("时间节点人数表格:")
    print("=" * 60)
    print(final_table.to_string(index=False))
    print(f"\n时间节点人数表格已保存至: {time_node_table_path}")

report_path = os.path.join(output_dir, "13_Subgroup_Analysis_Merged_Results.csv")
results_df.to_csv(report_path, index=False)

print("\n" + "=" * 60)
print("亚组分析结果汇总 (合并评分):")
print("=" * 60)

summary_cols = ['Dataset', 'Model', 'Scores', 'N', 'High_N', 'Low_N', 'HR_Str', 'P_Value_Str']
print(results_df[summary_cols].to_string(index=False))
print(f"\n详细结果已保存至: {report_path}")

# ==========================================
# 7. 绘图
# ==========================================

print("\n生成亚组生存曲线...")

for model_col in SUBGROUP_ANALYSIS_CONFIG.keys():
    if model_col not in train_df.columns:
        continue

    scores_to_plot = SUBGROUP_ANALYSIS_CONFIG[model_col]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    plt.subplots_adjust(wspace=0.35, bottom=0.25, top=0.9)

    for col_idx, dataset_name in enumerate(['Training', 'Test']):
        ax = axes[col_idx]
        current_df = train_df if dataset_name == 'Training' else test_df
        current_X = train_df_X if dataset_name == 'Training' else test_df_X

        risk_scores = gbs_model.predict(current_X)
        current_df['GBS_Risk'] = np.where(risk_scores >= optimal_threshold, 'High', 'Low')

        sub_df = current_df[current_df[model_col].isin(scores_to_plot)].copy()

        dataset_cn = "训练集" if dataset_name == 'Training' else "测试集"
        model_cn = {
            'Leibovich': 'Leibovich',
            'UISS': 'UISS',
            'KEYNOTE-564': 'KEYNOTE-564'
        }[model_col]

        base_title = f"{dataset_cn}({model_cn}低中危患者再分层)"

        if len(sub_df) == 0:
            ax.text(0.5, 0.5, 'No Data or Single Group', ha='center', va='center')
            ax.set_title(base_title, fontsize=28, fontweight='bold', pad=22)
            continue

        if len(sub_df['GBS_Risk'].unique()) < 2:
            ax.text(0.5, 0.5, 'No Data or Single Group', ha='center', va='center')
            ax.set_title(base_title, fontsize=28, fontweight='bold', pad=22)
            continue

        kmf = KaplanMeierFitter()

        low_df = sub_df[sub_df['GBS_Risk'] == 'Low']
        kmf.fit(low_df[time_col], low_df[event_col], label='低风险')
        kmf.plot_survival_function(ax=ax, color=color_low, lw=3)

        high_df = sub_df[sub_df['GBS_Risk'] == 'High']
        kmf.fit(high_df[time_col], high_df[event_col], label='高风险')
        kmf.plot_survival_function(ax=ax, color=color_high, lw=3)

        res_row = results_df[
            (results_df['Dataset'] == dataset_name) &
            (results_df['Model'] == model_col)
            ]

        if not res_row.empty:
            res = res_row.iloc[0]
            info_txt = ""
            if pd.notna(res['HR']):
                info_txt += f"HR={res['HR']:.2f} (95% CI: {res['HR_lower']:.2f}-{res['HR_upper']:.2f})\n"
            info_txt += f"P={format_p_val(res['P_value'])}"

            ax.text(0.05, 0.15, info_txt, transform=ax.transAxes, fontsize=16, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=1.5))

        ax.set_title(base_title, fontsize=28, fontweight='bold', pad=22)
        ax.set_xlabel("生存时间（月）", fontsize=24, fontweight='bold', labelpad=15)
        ax.set_ylabel("无病生存概率", fontsize=24, fontweight='bold', labelpad=15)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=22)

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
                    text_obj.set_position((x_coord, y_coord - 0.10))
                else:
                    text_obj.set_fontsize(16)

    plt.tight_layout()
    plot_filename = f"13_Subgroup_KM_{model_col}_Merged.png"
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f" 已保存: {plot_filename}")

print("\n" + "=" * 60)
print("所有亚组分析完成！")
print("=" * 60)
