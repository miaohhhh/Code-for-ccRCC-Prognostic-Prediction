from scipy.stats import mannwhitneyu, chi2_contingency

print("\n" + "=" * 50)
print("开始进行单因素特征初筛")
print("=" * 50)

group_0 = train_df[train_df['Recurrence'] == 0]
group_1 = train_df[train_df['Recurrence'] == 1]

features_to_screen = clinical_vars + radiomics_vars
print(f"待筛选特征总数: {len(features_to_screen)}")
print(f" - 临床特征: {len(clinical_vars)}")
print(f" - 影像组学特征: {len(radiomics_vars)}")

p_values = []
significant_features = []

for feature in features_to_screen:
    p_val = np.nan
    method_used = "Unknown"
    try:
        if feature in clinical_cat_vars:
            crosstab = pd.crosstab(train_df[feature], train_df['Recurrence'])
            chi2, p, dof, expected = chi2_contingency(crosstab)
            p_val = p
            method_used = "Chi2"
        else:
            data_0 = group_0[feature]
            data_1 = group_1[feature]
            u_stat, p = mannwhitneyu(data_0, data_1, alternative='two-sided')
            p_val = p
            method_used = "MWU"

        p_values.append({'Feature': feature, 'P_Value': p_val, 'Method': method_used})

        if p_val < 0.05:
            significant_features.append(feature)
    except Exception as e:
        print(f"特征 {feature} ({method_used}) 检验失败: {e}")
        p_values.append({'Feature': feature, 'P_Value': np.nan, 'Method': "Error"})

# ==========================================
# 9. 筛选结果汇总
# ==========================================

univariate_results_df = pd.DataFrame(p_values)
univariate_results_df = univariate_results_df.sort_values(by='P_Value')

uni_save_path = os.path.join(output_dir, "2_Univariate_Results_All.csv")
univariate_results_df.to_csv(uni_save_path, index=False)
print(f"单因素检验详细结果已保存至: {uni_save_path}")

print(f"\n原始特征总数: {len(features_to_screen)}")
print(f"经检验筛选后特征数 (P < 0.05): {len(significant_features)}")

selected_clinical = [f for f in significant_features if f in clinical_vars]
selected_radiomics = [f for f in significant_features if f in radiomics_vars]
print(f" - 筛选出的临床特征: {len(selected_clinical)}")
print(f" - 筛选出的影像组学特征: {len(selected_radiomics)}")

# ==========================================
# 10. 保存初筛后的特征列表
# ==========================================

features_list_path = os.path.join(output_dir, "2_Selected_Features_After_Uni.csv")
pd.DataFrame(significant_features, columns=['Selected_Features']).to_csv(features_list_path, index=False)

print("\n" + "=" * 50)
print("特征初筛完成！")
print("=" * 50)
print(f"最终用于下一步降维的特征总数: {len(significant_features)}")
print(f"特征列表已保存至: {features_list_path}")

if len(significant_features) > 0:
    print("\n显著性最高的前5个特征:")
    print(univariate_results_df.head())
