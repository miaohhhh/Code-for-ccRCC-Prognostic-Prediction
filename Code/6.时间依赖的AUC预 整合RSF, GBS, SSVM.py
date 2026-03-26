# ==========================================
# 生存分析建模与评估 (RSF, GBS, SSVM) - 抗过拟合优化版
# ==========================================
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import joblib
from datetime import datetime
from sklearn.metrics import roc_curve, auc, roc_auc_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 忽略警告
warnings.filterwarnings('ignore')

print("\n" + "=" * 60)
print("生存分析建模与评估 (RSF, GBS, SSVM) - 抗过拟合优化版")
print("=" * 60)

# ==========================================
# 1. 加载数据和特征
# ==========================================
# 路径设置
output_dir = r"D:\Renal Data\2017-2024 Data\Complete\fig"

# 读取标准化后的数据
train_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "2_test_data_scaled.csv"))

# 读取最终的特征列表
final_features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = final_features_df['Final_Feature_Name'].tolist()

print(f"训练集样本数: {len(train_df)}")
print(f"测试集样本数: {len(test_df)}")
print(f"最终特征数: {len(final_features)}")

# 检查生存时间列是否存在
# 【修改】PFS_time -> DFS_time, PFS_event -> DFS_event
if 'DFS_time' not in train_df.columns or 'DFS_event' not in train_df.columns:
    print("错误: 数据集中缺少生存时间或生存事件列")
    print(f"可用的列: {train_df.columns.tolist()[:20]}...")
    # 尝试寻找可能的列名
    for col in train_df.columns:
        if 'time' in col.lower() or 'surv' in col.lower():
            print(f"可能的时间列: {col}")
        if 'event' in col.lower() or 'recurrence' in col.lower() or 'status' in col.lower():
            print(f"可能的事件列: {col}")
    # 退出或使用默认值
    time_col = 'Time'  # 请根据实际情况修改
    event_col = 'Event'  # 请根据实际情况修改
else:
    time_col = 'DFS_time'
    event_col = 'DFS_event'
    print(f"使用生存时间列: {time_col}, 生存事件列: {event_col}")

# ==========================================
# 2. 准备生存分析数据
# ==========================================
# 创建生存分析所需的数据结构
X_train = train_df[final_features].copy()
X_test = test_df[final_features].copy()

# 提取生存时间和事件
T_train = train_df[time_col].values
E_train = train_df[event_col].values
T_test = test_df[time_col].values
E_test = test_df[event_col].values

print(f"\n生存时间统计:")
print(f"训练集 - 中位生存时间: {np.median(T_train):.1f} 月")
print(f"测试集 - 中位生存时间: {np.median(T_test):.1f} 月")
print(f"训练集事件率: {E_train.mean():.2%}")
print(f"测试集事件率: {E_test.mean():.2%}")

# 定义评估时间点（以月为单位）
evaluation_times = [36, 60]  # 3年=36个月，5年=60个月
print(f"\n评估时间点: {evaluation_times} 月 (3年, 5年)")

# 检查每个时间点的事件数
print("\n各时间点事件统计:")
for time_point in evaluation_times:
    train_events = ((T_train <= time_point) & (E_train == 1)).sum()
    test_events = ((T_test <= time_point) & (E_test == 1)).sum()
    train_at_risk = (T_train >= time_point).sum()
    test_at_risk = (T_test >= time_point).sum()
    print(
        f"{time_point // 12}年({time_point}月): 训练集事件数={train_events}, 风险样本数={train_at_risk}, 测试集事件数={test_events}, 风险样本数={test_at_risk}")


# ==========================================
# 3. 辅助函数定义
# ==========================================
def bootstrap_auc(y_true, y_score, n_bootstraps=1000):
    """使用Bootstrap计算AUC的95%置信区间"""
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan

    bootstrapped_scores = []
    for i in range(n_bootstraps):
        indices = resample(np.arange(len(y_true)), random_state=i)
        if len(np.unique(y_true[indices])) < 2:
            continue
        try:
            score = roc_auc_score(y_true[indices], y_score[indices])
            bootstrapped_scores.append(score)
        except:
            continue

    if bootstrapped_scores:
        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        ci_lower = np.percentile(sorted_scores, 2.5)
        ci_upper = np.percentile(sorted_scores, 97.5)
        return np.mean(sorted_scores), ci_lower, ci_upper
    else:
        return np.nan, np.nan, np.nan


def time_dependent_auc(model, X, T, E, time_point):
    """通用的时间依赖AUC计算函数"""
    try:
        # 获取风险分数
        if hasattr(model, 'predict_partial_hazard'):
            # COX模型 (保留此分支以防万一，但实际不再使用COX)
            risk_scores = model.predict_partial_hazard(X)
            if hasattr(risk_scores, 'values'):
                risk_scores = risk_scores.values
        elif hasattr(model, 'predict'):
            # scikit-survival模型
            risk_scores = model.predict(X)
        else:
            print(f"模型不支持预测方法")
            return np.nan, np.nan, np.nan

        # 创建二元标签：在时间点前发生事件 vs 时间点后仍存活
        y_true_binary = np.zeros(len(X))
        y_true_binary[(T <= time_point) & (E == 1)] = 1  # 事件组
        y_true_binary[(T > time_point)] = 0  # 对照组

        # 只保留在时间点仍处于风险中的个体或在时间点前发生事件的个体
        valid_mask = (T > time_point) | ((T <= time_point) & (E == 1))
        y_true_binary = y_true_binary[valid_mask]
        y_score_binary = risk_scores[valid_mask]

        # 检查是否有足够的事件和非事件
        n_events = np.sum(y_true_binary == 1)
        n_non_events = np.sum(y_true_binary == 0)

        if n_events == 0 or n_non_events == 0:
            # print(f"警告: 时间点{time_point}月, 事件数={n_events}, 非事件数={n_non_events}")
            return np.nan, np.nan, np.nan

        if len(np.unique(y_true_binary)) < 2:
            # print(f"警告: 时间点{time_point}月, 只有一个类别")
            return np.nan, np.nan, np.nan

        # 直接计算AUC
        try:
            auc_value = roc_auc_score(y_true_binary, y_score_binary)

            # 使用简化方法计算置信区间
            se = np.sqrt(auc_value * (1 - auc_value) / (n_events + n_non_events))
            ci_lower = auc_value - 1.96 * se
            ci_upper = auc_value + 1.96 * se

            # 确保置信区间在[0,1]范围内
            ci_lower = max(0, ci_lower)
            ci_upper = min(1, ci_upper)

            return auc_value, ci_lower, ci_upper
        except Exception as e:
            print(f"计算AUC时出错: {e}")
            return np.nan, np.nan, np.nan
    except Exception as e:
        print(f"计算时间依赖AUC时出错: {e}")
        import traceback
        traceback.print_exc()
        return np.nan, np.nan, np.nan


def prepare_survival_data(T, E):
    """准备scikit-survival所需的结构化数组"""
    structured_array = np.array(
        [(E[i], T[i]) for i in range(len(T))],
        dtype=[('event', 'bool'), ('time', 'float64')]
    )
    return structured_array


# ==========================================
# 4. 随机生存森林 (RSF) - 抗过拟合优化
# ==========================================
print("\n" + "=" * 40)
print("随机生存森林 (RSF) - 抗过拟合优化")
print("=" * 40)

try:
    # 检查是否安装scikit-survival
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.metrics import concordance_index_censored
        from sklearn.model_selection import ParameterGrid
    except ImportError:
        print("请先安装scikit-survival: pip install scikit-survival")
        raise

    # 准备结构化数组（scikit-survival要求的格式）
    y_train_struct = prepare_survival_data(T_train, E_train)
    y_test_struct = prepare_survival_data(T_test, E_test)

    # ==========================================
    # ✅ 优化：限制深度，增加 min_samples，防止过拟合
    # ==========================================
    print("使用抗过拟合参数进行网格搜索...")

    # 定义参数组合 - 强制限制深度和增加最小样本数
    param_combinations = [
        # 组合1: 浅树，高正则化
        {'n_estimators': 200, 'max_depth': 3, 'min_samples_split': 20, 'min_samples_leaf': 10, 'random_state': 42},
        # 组合2: 中等深度，中等正则化
        {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 6, 'min_samples_leaf': 3, 'random_state': 42},
        # 组合3: 深度控制，限制分裂
        {'n_estimators': 300, 'max_depth': 4, 'min_samples_split': 10, 'min_samples_leaf': 8, 'random_state': 42},
        # 组合4: 非常保守的参数
        {'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 30, 'min_samples_leaf': 15, 'random_state': 42},
    ]

    best_score = -np.inf
    best_params = None
    best_model = None

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for params in param_combinations:
        print(f"\n测试参数组合: {params}")
        fold_scores = []

        for train_idx, val_idx in kf.split(X_train):
            # 分割数据
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train_struct[train_idx]
            y_val_fold = y_train_struct[val_idx]

            # 创建并训练模型
            try:
                # 尝试创建一个不包含criterion参数的模型
                model_params = {k: v for k, v in params.items() if k != 'criterion'}
                rsf = RandomSurvivalForest(**model_params)
                rsf.fit(X_train_fold, y_train_fold)

                # 预测并计算C-index
                predictions = rsf.predict(X_val_fold)
                c_index = concordance_index_censored(
                    y_val_fold['event'], y_val_fold['time'], predictions
                )[0]
                fold_scores.append(c_index)
            except Exception as e:
                print(f"参数组合{params}训练失败: {e}")
                fold_scores.append(0)
                continue

        if fold_scores:
            avg_score = np.mean(fold_scores)
            print(f"平均C-index: {avg_score:.4f}")

            if avg_score > best_score:
                best_score = avg_score
                best_params = params
                # 用全部训练数据训练最佳模型
                try:
                    best_model_params = {k: v for k, v in best_params.items() if k != 'criterion'}
                    best_model = RandomSurvivalForest(**best_model_params)
                    best_model.fit(X_train, y_train_struct)
                except Exception as e:
                    print(f"训练最终模型失败: {e}")

    if best_model is None:
        print("\n所有参数组合都失败，使用默认参数...")
        best_model = RandomSurvivalForest(random_state=42)
        best_model.fit(X_train, y_train_struct)
        best_params = {'random_state': 42}
    else:
        print(f"\n最佳参数: {best_params}")
        print(f"最佳C-index: {best_score:.4f}")

    # 保存模型
    rsf_model_path = os.path.join(output_dir, "4_rsf_model.pkl")
    joblib.dump(best_model, rsf_model_path)
    print(f"RSF模型已保存至: {rsf_model_path}")

    # 预测
    rsf_train_risk = best_model.predict(X_train)
    rsf_test_risk = best_model.predict(X_test)

    # 计算C-index
    train_c_index = concordance_index_censored(
        y_train_struct['event'], y_train_struct['time'], rsf_train_risk
    )[0]
    test_c_index = concordance_index_censored(
        y_test_struct['event'], y_test_struct['time'], rsf_test_risk
    )[0]

    print(f"\n训练集C-index: {train_c_index:.4f}")
    print(f"测试集C-index: {test_c_index:.4f}")
    print(f"过拟合程度 (Train-Test Gap): {train_c_index - test_c_index:.4f}")

    # 计算时间依赖AUC
    rsf_results = {'Model': 'Random Survival Forest', 'Train_C_index': f"{train_c_index:.4f}",
                   'Test_C_index': f"{test_c_index:.4f}",
                   'Gap': f"{train_c_index - test_c_index:.4f}"}

    for time_point in evaluation_times:
        # 训练集
        train_auc, train_ci_lower, train_ci_upper = time_dependent_auc(
            best_model, X_train, T_train, E_train, time_point
        )

        # 测试集
        test_auc, test_ci_lower, test_ci_upper = time_dependent_auc(
            best_model, X_test, T_test, E_test, time_point
        )

        rsf_results[
            f'Train_AUC_{time_point}m'] = f"{train_auc:.3f} ({train_ci_lower:.3f}-{train_ci_upper:.3f})" if not np.isnan(
            train_auc) else "NaN"
        rsf_results[
            f'Test_AUC_{time_point}m'] = f"{test_auc:.3f} ({test_ci_lower:.3f}-{test_ci_upper:.3f})" if not np.isnan(
            test_auc) else "NaN"

        print(f"\n{time_point // 12}年({time_point}月) AUC:")
        if not np.isnan(train_auc):
            print(f"  训练集: {train_auc:.3f} (95%CI: {train_ci_lower:.3f}-{train_ci_upper:.3f})")
        else:
            print(f"  训练集: 无法计算")
        if not np.isnan(test_auc):
            print(f"  测试集: {test_auc:.3f} (95%CI: {test_ci_lower:.3f}-{test_ci_upper:.3f})")
        else:
            print(f"  测试集: 无法计算")

    # 保存结果
    rsf_results_df = pd.DataFrame([rsf_results])
    rsf_results_path = os.path.join(output_dir, "4_rsf_model_results.csv")
    rsf_results_df.to_csv(rsf_results_path, index=False)
    print(f"\nRSF模型结果已保存至: {rsf_results_path}")

except Exception as e:
    print(f"RSF模型训练失败: {e}")
    import traceback

    traceback.print_exc()
    rsf_results = {'Model': 'Random Survival Forest', 'Error': str(e)}

# ==========================================
# 5. 生存梯度提升 (GBS) - 抗过拟合优化
# ==========================================
print("\n" + "=" * 40)
print("生存梯度提升 (GBS) - 抗过拟合优化")
print("=" * 40)

try:
    # 检查是否安装scikit-survival
    try:
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    except ImportError:
        print("请先安装scikit-survival: pip install scikit-survival")
        raise

    # ==========================================
    # ✅ 优化：降低学习率，固定深度，增加 subsample 随机性
    # ==========================================
    param_grid_gbs = {
        'n_estimators': [200, 300],
        'learning_rate': [0.01, 0.005],  # 降低学习率，让模型收敛更慢、更稳
        'max_depth': [3],  # 固定为3，GBS中3层通常足够且不易过拟合
        'min_samples_split': [10, 20],  # 增加分裂所需样本数
        'subsample': [0.7, 0.8],  # 关键：随机选择70%-80%的样本训练，抗过拟合
        'random_state': [42]
    }

    # 初始化GBS
    gbs = GradientBoostingSurvivalAnalysis(random_state=42)

    # 使用简化的网格搜索
    grid_search_gbs = GridSearchCV(
        gbs, param_grid_gbs, cv=3, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )

    print("开始GBS网格搜索...")
    grid_search_gbs.fit(X_train, y_train_struct)

    print(f"最佳参数: {grid_search_gbs.best_params_}")
    print(f"最佳分数: {-grid_search_gbs.best_score_:.4f}")

    # 使用最佳参数训练模型
    gbs_best = grid_search_gbs.best_estimator_
    gbs_best.fit(X_train, y_train_struct)

    # 保存模型
    gbs_model_path = os.path.join(output_dir, "4_gbs_model.pkl")
    joblib.dump(gbs_best, gbs_model_path)
    print(f"GBS模型已保存至: {gbs_model_path}")

    # 预测
    gbs_train_risk = gbs_best.predict(X_train)
    gbs_test_risk = gbs_best.predict(X_test)

    # 计算C-index
    train_c_index = concordance_index_censored(
        y_train_struct['event'], y_train_struct['time'], gbs_train_risk
    )[0]
    test_c_index = concordance_index_censored(
        y_test_struct['event'], y_test_struct['time'], gbs_test_risk
    )[0]

    print(f"\n训练集C-index: {train_c_index:.4f}")
    print(f"测试集C-index: {test_c_index:.4f}")
    print(f"过拟合程度 (Train-Test Gap): {train_c_index - test_c_index:.4f}")

    # 计算时间依赖AUC
    gbs_results = {'Model': 'Gradient Boosting Survival', 'Train_C_index': f"{train_c_index:.4f}",
                   'Test_C_index': f"{test_c_index:.4f}",
                   'Gap': f"{train_c_index - test_c_index:.4f}"}

    for time_point in evaluation_times:
        # 训练集
        train_auc, train_ci_lower, train_ci_upper = time_dependent_auc(
            gbs_best, X_train, T_train, E_train, time_point
        )

        # 测试集
        test_auc, test_ci_lower, test_ci_upper = time_dependent_auc(
            gbs_best, X_test, T_test, E_test, time_point
        )

        gbs_results[
            f'Train_AUC_{time_point}m'] = f"{train_auc:.3f} ({train_ci_lower:.3f}-{train_ci_upper:.3f})" if not np.isnan(
            train_auc) else "NaN"
        gbs_results[
            f'Test_AUC_{time_point}m'] = f"{test_auc:.3f} ({test_ci_lower:.3f}-{test_ci_upper:.3f})" if not np.isnan(
            test_auc) else "NaN"

        print(f"\n{time_point // 12}年({time_point}月) AUC:")
        if not np.isnan(train_auc):
            print(f"  训练集: {train_auc:.3f} (95%CI: {train_ci_lower:.3f}-{train_ci_upper:.3f})")
        else:
            print(f"  训练集: 无法计算")
        if not np.isnan(test_auc):
            print(f"  测试集: {test_auc:.3f} (95%CI: {test_ci_lower:.3f}-{test_ci_upper:.3f})")
        else:
            print(f"  测试集: 无法计算")

    # 保存结果
    gbs_results_df = pd.DataFrame([gbs_results])
    gbs_results_path = os.path.join(output_dir, "4_gbs_model_results.csv")
    gbs_results_df.to_csv(gbs_results_path, index=False)
    print(f"\nGBS模型结果已保存至: {gbs_results_path}")

except Exception as e:
    print(f"GBS模型训练失败: {e}")
    import traceback

    traceback.print_exc()
    gbs_results = {'Model': 'Gradient Boosting Survival', 'Error': str(e)}

# ==========================================
# 6. 生存支持向量机 (SSVM) - 抗过拟合优化
# ==========================================
print("\n" + "=" * 40)
print("生存支持向量机 (SSVM) - 抗过拟合优化")
print("=" * 40)

try:
    # 检查是否安装必要的库
    try:
        # 尝试导入scikit-survival中的SSVM
        from sksurv.svm import FastSurvivalSVM
        from sksurv.metrics import concordance_index_censored

        print("使用scikit-survival中的FastSurvivalSVM")
        use_sksurv_svm = True
    except ImportError:
        print("scikit-survival未安装或没有SSVM实现，尝试使用其他方法...")
        use_sksurv_svm = False

    if use_sksurv_svm:
        # ==========================================
        # ✅ 优化：增大 alpha 正则化参数，增强正则化
        # ==========================================
        # 简化参数搜索
        simplified_params = [
            # 大幅增加正则化强度
            {'alpha': 1.0, 'rank_ratio': 1.0, 'max_iter': 1000, 'random_state': 42, 'tol': 1e-3},
            {'alpha': 5.0, 'rank_ratio': 1.0, 'max_iter': 1000, 'random_state': 42, 'tol': 1e-3},
            {'alpha': 10.0, 'rank_ratio': 0.8, 'max_iter': 1000, 'random_state': 42, 'tol': 1e-3},
            # 中等正则化
            {'alpha': 0.5, 'rank_ratio': 1.0, 'max_iter': 1000, 'random_state': 42, 'tol': 1e-3},
        ]

        best_score = -np.inf
        best_params = None
        best_model = None

        for params in simplified_params:
            try:
                ssvm_model = FastSurvivalSVM(**params)

                # 5折交叉验证评估
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                fold_scores = []

                for train_idx, val_idx in kf.split(X_train):
                    X_train_fold = X_train.iloc[train_idx]
                    X_val_fold = X_train.iloc[val_idx]
                    y_train_fold = y_train_struct[train_idx]
                    y_val_fold = y_train_struct[val_idx]

                    ssvm_model.fit(X_train_fold, y_train_fold)
                    predictions = ssvm_model.predict(X_val_fold)
                    c_index = concordance_index_censored(
                        y_val_fold['event'], y_val_fold['time'], predictions
                    )[0]
                    fold_scores.append(c_index)

                avg_score = np.mean(fold_scores)
                print(f"参数 {params}: 平均C-index = {avg_score:.4f}")

                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params
                    # 用全部训练数据训练模型
                    best_model = FastSurvivalSVM(**params)
                    best_model.fit(X_train, y_train_struct)

            except Exception as e:
                print(f"参数 {params} 训练失败: {e}")

        if best_model is None:
            print("所有参数组合都失败，使用默认参数...")
            best_model = FastSurvivalSVM(random_state=42)
            best_model.fit(X_train, y_train_struct)
            best_params = {'random_state': 42}
        else:
            print(f"\n最佳参数: {best_params}")
            print(f"最佳C-index: {best_score:.4f}")

        # 保存模型
        ssvm_model_path = os.path.join(output_dir, "4_ssvm_model.pkl")
        joblib.dump(best_model, ssvm_model_path)
        print(f"SSVM模型已保存至: {ssvm_model_path}")

        # 预测风险分数
        ssvm_train_risk = best_model.predict(X_train)
        ssvm_test_risk = best_model.predict(X_test)

        # 计算C-index
        train_c_index = concordance_index_censored(
            y_train_struct['event'], y_train_struct['time'], ssvm_train_risk
        )[0]
        test_c_index = concordance_index_censored(
            y_test_struct['event'], y_test_struct['time'], ssvm_test_risk
        )[0]

        print(f"\n训练集C-index: {train_c_index:.4f}")
        print(f"测试集C-index: {test_c_index:.4f}")
        print(f"过拟合程度 (Train-Test Gap): {train_c_index - test_c_index:.4f}")

        # 计算时间依赖AUC
        ssvm_results = {'Model': 'Survival SVM', 'Train_C_index': f"{train_c_index:.4f}",
                        'Test_C_index': f"{test_c_index:.4f}",
                        'Gap': f"{train_c_index - test_c_index:.4f}"}

        for time_point in evaluation_times:
            # 训练集
            train_auc, train_ci_lower, train_ci_upper = time_dependent_auc(
                best_model, X_train, T_train, E_train, time_point
            )

            # 测试集
            test_auc, test_ci_lower, test_ci_upper = time_dependent_auc(
                best_model, X_test, T_test, E_test, time_point
            )

            ssvm_results[
                f'Train_AUC_{time_point}m'] = f"{train_auc:.3f} ({train_ci_lower:.3f}-{train_ci_upper:.3f})" if not np.isnan(
                train_auc) else "NaN"
            ssvm_results[
                f'Test_AUC_{time_point}m'] = f"{test_auc:.3f} ({test_ci_lower:.3f}-{test_ci_upper:.3f})" if not np.isnan(
                test_auc) else "NaN"

            print(f"\n{time_point // 12}年({time_point}月) AUC:")
            if not np.isnan(train_auc):
                print(f"  训练集: {train_auc:.3f} (95%CI: {train_ci_lower:.3f}-{train_ci_upper:.3f})")
            else:
                print(f"  训练集: 无法计算")
            if not np.isnan(test_auc):
                print(f"  测试集: {test_auc:.3f} (95%CI: {test_ci_lower:.3f}-{test_ci_upper:.3f})")
            else:
                print(f"  测试集: 无法计算")

        # 保存结果
        ssvm_results_df = pd.DataFrame([ssvm_results])
        ssvm_results_path = os.path.join(output_dir, "4_ssvm_model_results.csv")
        ssvm_results_df.to_csv(ssvm_results_path, index=False)
        print(f"\nSSVM模型结果已保存至: {ssvm_results_path}")

    else:
        # 如果scikit-survival不可用，尝试使用基于转换的方法
        print("尝试使用基于转换的SSVM方法...")

        # 方法1: 将生存问题转换为分类问题（在特定时间点）
        def survival_to_classification(T, E, time_point):
            """将生存数据转换为特定时间点的二分类问题"""
            y_binary = np.zeros(len(T))
            # 在时间点前发生事件 = 1
            y_binary[(T <= time_point) & (E == 1)] = 1
            # 在时间点后仍存活 = 0
            y_binary[T > time_point] = 0
            # 删失且在时间点前：无法确定，通常排除
            mask = (T <= time_point) & (E == 0)
            y_binary = np.delete(y_binary, mask)
            return y_binary

        # 选择36个月作为转换时间点
        time_point = 36
        y_train_binary = survival_to_classification(T_train, E_train, time_point)
        # 对应地筛选特征
        mask_train = ~((T_train <= time_point) & (E_train == 0))
        X_train_binary = X_train[mask_train].copy()

        # 使用SVM进行分类
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV

        # 网格搜索参数 - 增加C值范围，增强正则化
        param_grid_svc = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'],
            'random_state': [42]
        }

        svc = SVC(probability=True)
        grid_search_svc = GridSearchCV(
            svc, param_grid_svc, cv=3, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )

        print("开始SVM分类器网格搜索...")
        grid_search_svc.fit(X_train_binary, y_train_binary)

        print(f"最佳参数: {grid_search_svc.best_params_}")
        print(f"最佳AUC: {grid_search_svc.best_score_:.4f}")

        # 使用最佳参数训练模型
        svc_best = grid_search_svc.best_estimator_

        # 保存模型
        ssvm_model_path = os.path.join(output_dir, "4_ssvm_svc_model.pkl")
        joblib.dump(svc_best, ssvm_model_path)
        print(f"SVM分类器模型已保存至: {ssvm_model_path}")

        # 对于测试集，同样进行转换
        y_test_binary = survival_to_classification(T_test, E_test, time_point)
        mask_test = ~((T_test <= time_point) & (E_test == 0))
        X_test_binary = X_test[mask_test].copy()

        # 预测概率
        y_train_pred = svc_best.predict_proba(X_train_binary)[:, 1]
        y_test_pred = svc_best.predict_proba(X_test_binary)[:, 1]

        # 计算AUC
        train_auc = roc_auc_score(y_train_binary, y_train_pred)
        test_auc = roc_auc_score(y_test_binary, y_test_pred)

        print(f"\n36个月时间点分类AUC:")
        print(f"训练集AUC: {train_auc:.4f}")
        print(f"测试集AUC: {test_auc:.4f}")

        # 创建结果字典
        ssvm_results = {'Model': 'Survival SVM (Binary at 36m)'}
        ssvm_results['Train_AUC_36m'] = f"{train_auc:.3f}"
        ssvm_results['Test_AUC_36m'] = f"{test_auc:.3f}"
        # 由于是分类模型，C-index不适用
        ssvm_results['Train_C_index'] = "N/A"
        ssvm_results['Test_C_index'] = "N/A"

        # 保存结果
        ssvm_results_df = pd.DataFrame([ssvm_results])
        ssvm_results_path = os.path.join(output_dir, "4_ssvm_model_results.csv")
        ssvm_results_df.to_csv(ssvm_results_path, index=False)
        print(f"\nSSVM模型结果已保存至: {ssvm_results_path}")

except Exception as e:
    print(f"SSVM模型训练失败: {e}")
    import traceback

    traceback.print_exc()
    ssvm_results = {'Model': 'Survival SVM', 'Error': str(e)}

# ==========================================
# 7. 结果汇总与可视化
# ==========================================
print("\n" + "=" * 60)
print("模型性能汇总")
print("=" * 60)

# 收集所有结果
all_results = []

# 【修改】尝试读取每个模型的结果，顺序调整为 GBS, RSF, SSVM
model_files = {
    'Gradient Boosting Survival': '4_gbs_model_results.csv',
    'Random Survival Forest': '4_rsf_model_results.csv',
    'Survival SVM': '4_ssvm_model_results.csv'
}

for model_name, file_name in model_files.items():
    file_path = os.path.join(output_dir, file_name)
    if os.path.exists(file_path):
        try:
            results_df = pd.read_csv(file_path)
            all_results.append(results_df)
        except:
            print(f"无法读取 {file_path}")

# 合并结果
if all_results:
    summary_df = pd.concat(all_results, ignore_index=True)
    summary_path = os.path.join(output_dir, "4_model_comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n模型比较结果已保存至: {summary_path}")
    print("\n模型性能汇总:")
    print(summary_df.to_string(index=False))
else:
    print("没有可用的模型结果进行汇总")

# ==========================================
# 8. 绘制时间依赖ROC曲线
# ==========================================
print("\n" + "=" * 60)
print("绘制时间依赖ROC曲线")
print("=" * 60)

def plot_simple_roc_curve(model, X, T, E, time_point, model_name, ax, set_name):
    """绘制单个模型的ROC曲线"""
    try:
        # 获取风险分数
        if hasattr(model, 'predict_partial_hazard'):
            risk_scores = model.predict_partial_hazard(X)
            if hasattr(risk_scores, 'values'):
                risk_scores = risk_scores.values
        elif hasattr(model, 'predict'):
            risk_scores = model.predict(X)
        else:
            return

        # 创建二元标签
        y_true_binary = np.zeros(len(X))
        y_true_binary[(T <= time_point) & (E == 1)] = 1
        y_true_binary[(T > time_point)] = 0

        # 只保留有效样本
        valid_mask = (T > time_point) | ((T <= time_point) & (E == 1))
        y_true_binary = y_true_binary[valid_mask]
        y_score_binary = risk_scores[valid_mask]

        # 检查是否有足够的事件和非事件
        if len(np.unique(y_true_binary)) < 2:
            return

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
        roc_auc = auc(fpr, tpr)

        # 绘制曲线
        ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    except Exception as e:
        print(f"绘制{model_name} ROC曲线时出错: {e}")

# 【修改】尝试加载所有模型，顺序调整为 GBS, RSF, SSVM
models_to_plot = {}

model_paths = {
    'GBS': '4_gbs_model.pkl',
    'RSF': '4_rsf_model.pkl',
    'SSVM': '4_ssvm_model.pkl'
}

for model_name, file_name in model_paths.items():
    file_path = os.path.join(output_dir, file_name)
    if os.path.exists(file_path):
        try:
            model = joblib.load(file_path)
            models_to_plot[model_name] = model
            print(f"已加载模型: {model_name}")
        except:
            print(f"无法加载模型: {model_name}")

# 为每个时间点绘制ROC曲线
if models_to_plot:
    for time_point in evaluation_times:
        # 训练集ROC曲线
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for model_name, model in models_to_plot.items():
            plot_simple_roc_curve(model, X_train, T_train, E_train, time_point, model_name, ax1, "Train")

        ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('假阳性率 (1-特异性)')
        ax1.set_ylabel('真阳性率 (敏感性)')
        ax1.set_title(f'{time_point // 12}年({time_point}月)时间依赖ROC曲线 - 训练集')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # 测试集ROC曲线
        for model_name, model in models_to_plot.items():
            plot_simple_roc_curve(model, X_test, T_test, E_test, time_point, model_name, ax2, "Test")

        ax2.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('假阳性率 (1-特异性)')
        ax2.set_ylabel('真阳性率 (敏感性)')
        ax2.set_title(f'{time_point // 12}年({time_point}月)时间依赖ROC曲线 - 测试集')
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"4_time_dependent_roc_{time_point}m.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC曲线已保存至: {plot_path}")
else:
    print("没有可用的模型用于绘制ROC曲线")

# ==========================================
# 9. 生成最终报告
# ==========================================
print("\n" + "=" * 60)
print("生存分析建模完成！")
print("=" * 60)
print("已完成的步骤:")
print("1. 梯度提升生存(GBS)模型训练与评估")
print("2. 随机生存森林(RSF)模型训练与评估")
print("3. 生存支持向量机(SSVM)模型训练与评估")
print("4. 时间依赖ROC曲线计算与可视化")
print("5. 模型性能汇总与比较")
print(f"\n所有结果已保存至: {output_dir}")

# 创建简单的文本报告
report_path = os.path.join(output_dir, "4_survival_analysis_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("生存分析建模报告 - 抗过拟合优化版 (GBS, RSF, SSVM)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"数据统计:\n")
    f.write(f"  训练集样本数: {len(train_df)}\n")
    f.write(f"  测试集样本数: {len(test_df)}\n")
    f.write(f"  最终特征数: {len(final_features)}\n\n")
    f.write(f"生存时间统计:\n")
    f.write(f"  训练集 - 中位生存时间: {np.median(T_train):.1f} 月\n")
    f.write(f"  测试集 - 中位生存时间: {np.median(T_test):.1f} 月\n")
    f.write(f"  训练集事件率: {E_train.mean():.2%}\n")
    f.write(f"  测试集事件率: {E_test.mean():.2%}\n\n")
    f.write(f"各时间点事件统计:\n")
    for time_point in evaluation_times:
        train_events = ((T_train <= time_point) & (E_train == 1)).sum()
        test_events = ((T_test <= time_point) & (E_test == 1)).sum()
        train_at_risk = (T_train >= time_point).sum()
        test_at_risk = (T_test >= time_point).sum()
        f.write(
            f"  {time_point // 12}年({time_point}月): 训练集事件数={train_events}, 风险样本数={train_at_risk}, 测试集事件数={test_events}, 风险样本数={test_at_risk}\n")

    f.write("\n" + "=" * 60 + "\n")
    f.write("模型性能汇总\n")
    f.write("=" * 60 + "\n\n")

    f.write("优化说明：\n")
    f.write("  - RSF: 增加min_samples_leaf, 限制max_depth\n")
    f.write("  - GBS: 降低learning_rate, 增加subsample, 固定max_depth=3\n")
    f.write("  - SSVM: 增大alpha正则化参数\n\n")

    if all_results:
        for idx, row in summary_df.iterrows():
            f.write(f"{row['Model']}:\n")
            for col in summary_df.columns:
                if col != 'Model':
                    f.write(f"  {col}: {row[col]}\n")
            f.write("\n")

    f.write("\n过拟合程度评估：\n")
    f.write("  - Gap = Train_C_index - Test_C_index\n")
    f.write("  - Gap < 0.05: 过拟合较轻，泛化能力好\n")
    f.write("  - 0.05 <= Gap < 0.10: 存在一定过拟合\n")
    f.write("  - Gap >= 0.10: 过拟合较严重，需进一步调整\n")

print(f"\n详细报告已保存至: {report_path}")
