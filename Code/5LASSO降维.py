import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("\n" + "=" * 50)
print("步骤 1: 去除高度共线性特征...")
print("=" * 50)

print(f"当前待降维特征数 (含临床+影像): {len(significant_features)}")

corr_matrix = train_df[significant_features].corr().abs()

plt.figure(figsize=(12, 10))
plot_features = significant_features[:50]
sns.heatmap(train_df[plot_features].corr().abs(), cmap='coolwarm', center=0, xticklabels=True, yticklabels=True)
plt.title("Correlation Matrix of Top 50 Features (Clinical + Radiomics)")
heatmap_path = os.path.join(output_dir, "3_Correlation_Heatmap_Example.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"相关性热图示例已保存 (前50个特征): {heatmap_path}")

corr_threshold = 0.9

upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_threshold)]

features_no_corr = [f for f in significant_features if f not in to_drop]

print(f"因高相关性剔除的特征数 (r > {corr_threshold}): {len(to_drop)}")
print(f"去重后剩余特征数: {len(features_no_corr)}")

pd.DataFrame(features_no_corr, columns=['Uncorrelated_Features']).to_csv(
    os.path.join(output_dir, "3_Uncorrelated_Features.csv"),
    index=False
)

# ==========================================
# 12. 特征降维 
# ==========================================

from sklearn.linear_model import LassoCV, lasso_path

print("\n" + "=" * 50)
print("步骤 2: LASSO 回归特征选择...")
print("=" * 50)

X_train_lasso = train_df[features_no_corr]
y_train_lasso = train_df['Recurrence']

lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1)
lasso_cv.fit(X_train_lasso, y_train_lasso)


lasso_coefs = pd.Series(lasso_cv.coef_, index=X_train_lasso.columns)
selected_lasso_features = lasso_coefs[lasso_coefs != 0].index.tolist()

print(f"\n经 LASSO 选定的关键特征数: {len(selected_lasso_features)}")

final_clinical_count = len([f for f in selected_lasso_features if f in clinical_vars])
final_rad_count = len([f for f in selected_lasso_features if f in radiomics_vars])
print(f" - 临床特征: {final_clinical_count}")
print(f" - 影像组学特征: {final_rad_count}")

# ==========================================
# 13. 结果可视化与保存 
# ==========================================

if len(selected_lasso_features) > 0:
    final_lasso_results = pd.DataFrame({
        'Feature': selected_lasso_features,
        'Coefficient': lasso_coefs[selected_lasso_features]
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(final_lasso_results['Feature'], final_lasso_results['Coefficient'], color='lightblue')
    plt.xticks(rotation=60, ha='right')
    plt.ylabel("Coefficient")
    plt.title("Lasso Regression Coefficients")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lasso_coefficients_bar.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.semilogx(lasso_cv.alphas_, lasso_cv.mse_path_.mean(axis=1), marker='o', color='red')
    plt.axvline(lasso_cv.alpha_, color='black', linestyle='--', label=f'Optimal Alpha: {lasso_cv.alpha_:.6f}')
    plt.xlabel("Lambda (Alpha)")
    plt.ylabel("MSE")
    plt.title("MSE vs Lambda (LassoCV Cross-Validation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mse_vs_lambda.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    alphas, coefs, _ = lasso_path(X_train_lasso, y_train_lasso)
    plt.semilogx(alphas, coefs.T)
    plt.axvline(lasso_cv.alpha_, color='black', linestyle='--', label=f'Optimal Alpha: {lasso_cv.alpha_:.6f}')
    plt.xlabel("Lambda (Alpha)")
    plt.ylabel("Coefficients")
    plt.title("Lasso Coefficient Paths")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lasso_coefficient_paths.png"), dpi=300)
    plt.close()

    lasso_result_path = os.path.join(output_dir, "3_LASSO_Selected_Features.csv")
    final_lasso_results.to_csv(lasso_result_path, index=False)
    print(f"LASSO 筛选结果及系数已保存至: {lasso_result_path}")
    print("\nLASSO 筛选出的 Top 10 关键特征 (按系数绝对值排序):")
    print(final_lasso_results.head(10))
else:
    print("\n警告: LASSO 未筛选出任何特征。请检查数据或调整参数。")
    final_lasso_results = pd.DataFrame(columns=['Feature', 'Coefficient'])

# ==========================================
# 14. 构建最终建模特征列表
# ==========================================

final_features_for_modeling = selected_lasso_features

print("\n" + "=" * 50)
print("降维流程结束！")
print("=" * 50)
print(f"最终特征总数: {len(final_features_for_modeling)}")

final_list_path = os.path.join(output_dir, "3_Final_Features_For_Modeling.csv")
pd.DataFrame(final_features_for_modeling, columns=['Final_Feature_Name']).to_csv(final_list_path, index=False)
print(f"最终建模特征列表已保存至: {final_list_path}")
