from sklearn.preprocessing import StandardScaler
import joblib

print("\n" + "="*50)
print("开始进行数据标准化...")
print("="*50)

continuous_features = clinical_cont_vars + radiomics_vars
print(f"需要进行标准化的连续变量数量: {len(continuous_features)}")
print(f" - 临床连续变量: {clinical_cont_vars}")
print(f" - 影像组学特征: {len(radiomics_vars)}")

scaler = StandardScaler()
scaler.fit(train_df[continuous_features])

train_df[continuous_features] = scaler.transform(train_df[continuous_features])
test_df[continuous_features] = scaler.transform(test_df[continuous_features])

print("标准化完成。")

scaler_save_path = os.path.join(output_dir, "2_standard_scaler.pkl")
joblib.dump(scaler, scaler_save_path)
print(f"标准化模型已保存至: {scaler_save_path}")

train_scaled_path = os.path.join(output_dir, "2_train_data_scaled.csv")
test_scaled_path = os.path.join(output_dir, "2_test_data_scaled.csv")
train_df.to_csv(train_scaled_path, index=False)
test_df.to_csv(test_scaled_path, index=False)
print(f"标准化后的数据已保存 (训练集: {train_scaled_path}, 测试集: {test_scaled_path})")
