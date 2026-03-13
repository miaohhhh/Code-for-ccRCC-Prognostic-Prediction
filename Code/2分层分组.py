from sklearn.model_selection import train_test_split

random_seed = 42

train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=random_seed,
    stratify=df['Recurrence']
)

print(f"随机种子: {random_seed}")

# ==========================================
# 5. 分层效果验证
# ==========================================

print("\n" + "=" * 50)
print("分层验证 (Recurrence 分布对比):")
print("=" * 50)

def check_recurrence_ratio(dataframe, set_name):
    total = len(dataframe)
    ratio = dataframe['Recurrence'].value_counts(normalize=True)
    count = dataframe['Recurrence'].value_counts()
    print(f"\n{set_name} (样本量: {total}):")
    for status in sorted(ratio.index):
        print(f" Status {status}: {ratio[status]:.2%} (数量: {count[status]})")

check_recurrence_ratio(df, "原始数据集")
check_recurrence_ratio(train_df, "训练集")
check_recurrence_ratio(test_df, "测试集")

# ==========================================
# 6. 保存划分后的数据
# ==========================================

train_save_path = os.path.join(output_dir, "train_data.csv")
test_save_path = os.path.join(output_dir, "test_data.csv")

train_df.to_csv(train_save_path, index=False)
test_df.to_csv(test_save_path, index=False)

print("\n" + "=" * 50)
print("文件保存成功:")
print(f"训练集: {train_save_path}")
print(f"测试集: {test_save_path}")
print("=" * 50)
