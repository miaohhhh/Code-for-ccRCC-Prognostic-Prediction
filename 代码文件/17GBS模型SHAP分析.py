import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams
from textwrap import wrap

rcParams['font.sans-serif'] = ['SimSun']
rcParams['axes.unicode_minus'] = False

print("\n" + "=" * 60)
print("GBS 模型SHAP分析 (修复SHAP计算问题)")
print("=" * 60)

# ==========================================
# 1. 配置参数
# ==========================================

time_col = 'PFS_time'
event_col = 'PFS_event'

# ==========================================
# 2. 加载数据与模型
# ==========================================

print("\n加载数据与模型...")

test_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))

features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = features_df['Final_Feature_Name'].tolist()

X_test = test_df[final_features]

try:
    gbs_model = joblib.load(os.path.join(output_dir, "4_gbs_model.pkl"))
    print(" ✓ GBS 模型加载成功")
except Exception as e:
    print(f" ✗ GBS 模型加载失败: {e}")
    exit()

# ==========================================
# 3. 计算SHAP值 
# ==========================================

print("\n计算SHAP值（这可能需要几分钟时间，取决于特征数量和样本大小）...")

shap_importance = None

try:
    explainer = shap.TreeExplainer(gbs_model)
    shap_values = explainer.shap_values(X_test)
    shap_importance = np.abs(shap_values).mean(axis=0)
    print(" ✓ SHAP值计算完成（使用TreeExplainer）")
except Exception as e:
    print(f" ⚠️ TreeExplainer计算失败: {e}")
    print(" 尝试使用KernelExplainer（速度较慢，但兼容性更好）...")
    try:
        sample_size = min(100, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42)
        explainer = shap.KernelExplainer(gbs_model.predict, X_sample)
        shap_values = explainer.shap_values(X_test, nsamples=100)
        shap_importance = np.abs(shap_values).mean(axis=0)
        print(" ✓ SHAP值计算完成（使用KernelExplainer）")
    except Exception as e:
        print(f" ✗ KernelExplainer计算也失败: {e}")
        print(" SHAP分析无法继续，请检查模型类型或安装shap库")
        exit()

if shap_importance is None:
    print(" ✗ SHAP值计算失败，无法进行特征重要性分析")
    exit()

# ==========================================
# 4. 提取贡献度前10的特征
# ==========================================

feature_importance = pd.DataFrame({
    'Feature': final_features,
    'SHAP_Importance': shap_importance
})

top_10_features = feature_importance.sort_values(by='SHAP_Importance', ascending=False).head(10)
top_10_feature_names = top_10_features['Feature'].tolist()

print("\n贡献度前10的特征:")
print(top_10_features.to_string(index=False))

top_features_path = os.path.join(output_dir, "16_GBS_SHAP_Top10_Features.csv")
top_10_features.to_csv(top_features_path, index=False)
print(f"\n前10特征已保存至: {top_features_path}")

top_10_indices = [final_features.index(feature) for feature in top_10_feature_names]
top_10_shap_values = shap_values[:, top_10_indices]

# ==========================================
# 5. 生成SHAP摘要图
# ==========================================

print("\n生成SHAP摘要图（前10特征，点图风格 - 中文标签，特征变量字体不变 + 调整字体大小）...")

plt.figure(figsize=(12, 10))

shap.summary_plot(
    top_10_shap_values,
    X_test[top_10_feature_names],
    feature_names=top_10_feature_names,
    plot_type="dot",
    show=False,
    color_bar=True
)

ax = plt.gca()

plt.xlabel('SHAP值', fontsize=14, fontweight='bold', labelpad=15)
plt.ylabel('特征', fontsize=14, fontweight='bold', labelpad=15)

y_labels = ax.get_yticklabels()
new_y_labels = []
for label in y_labels:
    text = label.get_text()
    wrapped_text = '\n'.join(wrap(text, width=20))
    new_y_labels.append(wrapped_text)

ax.set_yticklabels(new_y_labels, fontsize=11)
ax.tick_params(axis='x', labelsize=12)

fig = plt.gcf()
for current_ax in fig.axes:
    if current_ax != ax:
        if current_ax.get_ylabel():
            current_ax.set_ylabel('特征值', fontsize=12, labelpad=15)
            yticklabels = current_ax.get_yticklabels()
            new_yticklabels = []
            for label in yticklabels:
                text = label.get_text()
                if 'low' in text.lower() or text == 'Low':
                    text = '低'
                elif 'high' in text.lower() or text == 'High':
                    text = '高'
                new_yticklabels.append(text)
            current_ax.set_yticklabels(new_yticklabels, fontsize=10)

plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)

plot_path = os.path.join(output_dir, "16_GBS_SHAP_Top10_Summary_Plot_Wrapped.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"SHAP摘要图（调整纵坐标换行 + 调整字体大小）已保存至: {plot_path}")

print("\n" + "=" * 60)
print("SHAP分析完成！")
print("=" * 60)
