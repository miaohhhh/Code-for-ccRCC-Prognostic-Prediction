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

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')


# ==========================================
# 1. 加载数据和特征
# ==========================================

output_dir = r""

train_df = pd.read_csv(os.path.join(output_dir, "2_train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(output_dir, "2_test_data_scaled.csv"))

final_features_df = pd.read_csv(os.path.join(output_dir, "3_Final_Features_For_Modeling.csv"))
final_features = final_features_df['Final_Feature_Name'].tolist()

print(f"训练集样本数: {len(train_df)}")
print(f"测试集样本数: {len(test_df)}")
print(f"最终特征数: {len(final_features)}")

if 'DFS_time' not in train_df.columns or 'DFS_event' not in train_df.columns:
    print("错误: 数据集中缺少生存时间或生存事件列")
    print(f"可用的列: {train_df.columns.tolist()[:20]}...")
    for col in train_df.columns:
        if 'time' in col.lower() or 'surv' in col.lower():
            print(f"可能的时间列: {col}")
        if 'event' in col.lower() or 'recurrence' in col.lower() or 'status' in col.lower():
            print(f"可能的事件列: {col}")
    time_col = 'Time'
    event_col = 'Event'
else:
    time_col = 'DFS_time'
    event_col = 'DFS_event'
    print(f"使用生存时间列: {time_col}, 生存事件列: {event_col}")

# ==========================================
# 2. 准备生存分析数据
# ==========================================

X_train = train_df[final_features].copy()
X_test = test_df[final_features].copy()

T_train = train_df[time_col].values
E_train = train_df[event_col].values
T_test = test_df[time_col].values
E_test = test_df[event_col].values

print(f"\n生存时间统计:")
print(f"训练集 - 中位生存时间: {np.median(T_train):.1f} 月")
print(f"测试集 - 中位生存时间: {np.median(T_test):.1f} 月")
print(f"训练集事件率: {E_train.mean():.2%}")
print(f"测试集事件率: {E_test.mean():.2%}")

evaluation_times = [36, 60]

print(f"\n评估时间点: {evaluation_times} 月 (3年, 5年)")

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
    try:
        if hasattr(model, 'predict_partial_hazard'):
            risk_scores = model.predict_partial_hazard(X)
            if hasattr(risk_scores, 'values'):
                risk_scores = risk_scores.values
        elif hasattr(model, 'predict'):
            risk_scores = model.predict(X)
        else:
            print(f"模型不支持预测方法")
            return np.nan, np.nan, np.nan

        y_true_binary = np.zeros(len(X))
        y_true_binary[(T <= time_point) & (E == 1)] = 1
        y_true_binary[(T > time_point)] = 0

        valid_mask = (T > time_point) | ((T <= time_point) & (E == 1))
        y_true_binary = y_true_binary[valid_mask]
        y_score_binary = risk_scores[valid_mask]

        n_events = np.sum(y_true_binary == 1)
        n_non_events = np.sum(y_true_binary == 0)

        if n_events == 0 or n_non_events == 0:
            return np.nan, np.nan, np.nan

        if len(np.unique(y_true_binary)) < 2:
            return np.nan, np.nan, np.nan

        try:
            auc_value = roc_auc_score(y_true_binary, y_score_binary)
            se = np.sqrt(auc_value * (1 - auc_value) / (n_events + n_non_events))
            ci_lower = auc_value - 1.96 * se
            ci_upper = auc_value + 1.96 * se
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
    structured_array = np.array(
        [(E[i], T[i]) for i in range(len(T))],
        dtype=[('event', 'bool'), ('time', 'float64')]
    )
    return structured_array


# ==========================================
# 4. 随机生存森林 (RSF)
# ==========================================

try:
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.metrics import concordance_index_censored
        from sklearn.model_selection import ParameterGrid
    except ImportError:
        print("请先安装scikit-survival: pip install scikit-survival")
        raise

    y_train_struct = prepare_survival_data(T_train, E_train)
    y_test_struct = prepare_survival_data(T_test, E_test)


    param_combinations = [
        {'n_estimators': 200, 'max_depth': 3, 'min_samples_split': 20, 'min_samples_leaf': 10, 'random_state': 42},
        {'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 15, 'min_samples_leaf': 5, 'random_state': 42},
        {'n_estimators': 300, 'max_depth': 4, 'min_samples_split': 10, 'min_samples_leaf': 8, 'random_state': 42},
        {'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 30, 'min_samples_leaf': 15, 'random_state': 42},
    ]

    best_score = -np.inf
    best_params = None
    best_model = None

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for params in param_combinations:
        print(f"\n测试参数组合: {params}")
        fold_scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train_struct[train_idx]
            y_val_fold = y_train_struct[val_idx]

            try:
                model_params = {k: v for k, v in params.items() if k != 'criterion'}
                rsf = RandomSurvivalForest(**model_params)
                rsf.fit(X_train_fold, y_train_fold)

                predictions = rsf.predict(X_val_fold)
                c_index = concordance_index_censored(
                    y_val_fold['event'],
                    y_val_fold['time'],
                    predictions
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

    rsf_model_path = os.path.join(output_dir, "4_rsf_model.pkl")
    joblib.dump(best_model, rsf_model_path)
    print(f"RSF模型已保存至: {rsf_model_path}")

    rsf_train_risk = best_model.predict(X_train)
    rsf_test_risk = best_model.predict(X_test)

    train_c_index = concordance_index_censored(
        y_train_struct['event'],
        y_train_struct['time'],
        rsf_train_risk
    )[0]
    test_c_index = concordance_index_censored(
        y_test_struct['event'],
        y_test_struct['time'],
        rsf_test_risk
    )[0]

    print(f"\n训练集C-index: {train_c_index:.4f}")
    print(f"测试集C-index: {test_c_index:.4f}")
    print(f"过拟合程度 (Train-Test Gap): {train_c_index - test_c_index:.4f}")

    rsf_results = {'Model': 'Random Survival Forest',
                   'Train_C_index': f"{train_c_index:.4f}",
                   'Test_C_index': f"{test_c_index:.4f}",
                   'Gap': f"{train_c_index - test_c_index:.4f}"}

    for time_point in evaluation_times:
        train_auc, train_ci_lower, train_ci_upper = time_dependent_auc(
            best_model, X_train, T_train, E_train, time_point
        )
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
            print(f" 训练集: {train_auc:.3f} (95%CI: {train_ci_lower:.3f}-{train_ci_upper:.3f})")
        else:
            print(f" 训练集: 无法计算")
        if not np.isnan(test_auc):
            print(f" 测试集: {test_auc:.3f} (95%CI: {test_ci_lower:.3f}-{test_ci_upper:.3f})")
        else:
            print(f" 测试集: 无法计算")

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
# 5. 生存梯度提升 (GBS)
# ==========================================

try:
    try:
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    except ImportError:
        print("请先安装scikit-survival: pip install scikit-survival")
        raise

    param_grid_gbs = {
        'n_estimators': [200, 300],
        'learning_rate': [0.01, 0.005],
        'max_depth': [3],
        'min_samples_split': [10, 20],
        'subsample': [0.7, 0.8],
        'random_state': [42]
    }

    gbs = GradientBoostingSurvivalAnalysis(random_state=42)

    grid_search_gbs = GridSearchCV(
        gbs,
        param_grid_gbs,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    print("开始GBS网格搜索...")
    grid_search_gbs.fit(X_train, y_train_struct)

    print(f"最佳参数: {grid_search_gbs.best_params_}")
    print(f"最佳分数: {-grid_search_gbs.best_score_:.4f}")

    gbs_best = grid_search_gbs.best_estimator_
    gbs_best.fit(X_train, y_train_struct)

    gbs_model_path = os.path.join(output_dir, "4_gbs_model.pkl")
    joblib.dump(gbs_best, gbs_model_path)
    print(f"GBS模型已保存至: {gbs_model_path}")

    gbs_train_risk = gbs_best.predict(X_train)
    gbs_test_risk = gbs_best.predict(X_test)

    train_c_index = concordance_index_censored(
        y_train_struct['event'],
        y_train_struct['time'],
        gbs_train_risk
    )[0]
    test_c_index = concordance_index_censored(
        y_test_struct['event'],
        y_test_struct['time'],
        gbs_test_risk
    )[0]

    print(f"\n训练集C-index: {train_c_index:.4f}")
    print(f"测试集C-index: {test_c_index:.4f}")
    print(f"过拟合程度 (Train-Test Gap): {train_c_index - test_c_index:.4f}")

    gbs_results = {'Model': 'Gradient Boosting Survival',
                   'Train_C_index': f"{train_c_index:.4f}",
                   'Test_C_index': f"{test_c_index:.4f}",
                   'Gap': f"{train_c_index - test_c_index:.4f}"}

    for time_point in evaluation_times:
        train_auc, train_ci_lower, train_ci_upper = time_dependent_auc(
            gbs_best, X_train, T_train, E_train, time_point
        )
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
            print(f" 训练集: {train_auc:.3f} (95%CI: {train_ci_lower:.3f}-{train_ci_upper:.3f})")
        else:
            print(f" 训练集: 无法计算")
        if not np.isnan(test_auc):
            print(f" 测试集: {test_auc:.3f} (95%CI: {test_ci_lower:.3f}-{test_ci_upper:.3f})")
        else:
            print(f" 测试集: 无法计算")

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
# 6. 生存支持向量机 (SSVM)
# ==========================================

try:
    try:
        from sksurv.svm import FastSurvivalSVM
        from sksurv.metrics import concordance_index_censored

        print("使用scikit-survival中的FastSurvivalSVM")
        use_sksurv_svm = True
    except ImportError:
        print("scikit-survival未安装或没有SSVM实现，尝试使用其他方法...")
        use_sksurv_svm = False

    if use_sksurv_svm:
        simplified_params = [
            {'alpha': 1.0, 'rank_ratio': 1.0, 'max_iter': 1000, 'random_state': 42, 'tol': 1e-3},
            {'alpha': 5.0, 'rank_ratio': 1.0, 'max_iter': 1000, 'random_state': 42, 'tol': 1e-3},
            {'alpha': 10.0, 'rank_ratio': 0.8, 'max_iter': 1000, 'random_state': 42, 'tol': 1e-3},
            {'alpha': 0.5, 'rank_ratio': 1.0, 'max_iter': 1000, 'random_state': 42, 'tol': 1e-3},
        ]

        best_score = -np.inf
        best_params = None
        best_model = None

        for params in simplified_params:
            try:
                ssvm_model = FastSurvivalSVM(**params)

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
                        y_val_fold['event'],
                        y_val_fold['time'],
                        predictions
                    )[0]
                    fold_scores.append(c_index)

                avg_score = np.mean(fold_scores)
                print(f"参数 {params}: 平均C-index = {avg_score:.4f}")

                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params
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

        ssvm_model_path = os.path.join(output_dir, "4_ssvm_model.pkl")
        joblib.dump(best_model, ssvm_model_path)
        print(f"SSVM模型已保存至: {ssvm_model_path}")

        ssvm_train_risk = best_model.predict(X_train)
        ssvm_test_risk = best_model.predict(X_test)

        train_c_index = concordance_index_censored(
            y_train_struct['event'],
            y_train_struct['time'],
            ssvm_train_risk
        )[0]
        test_c_index = concordance_index_censored(
            y_test_struct['event'],
            y_test_struct['time'],
            ssvm_test_risk
        )[0]

        print(f"\n训练集C-index: {train_c_index:.4f}")
        print(f"测试集C-index: {test_c_index:.4f}")
        print(f"过拟合程度 (Train-Test Gap): {train_c_index - test_c_index:.4f}")

        ssvm_results = {'Model': 'Survival SVM',
                        'Train_C_index': f"{train_c_index:.4f}",
                        'Test_C_index': f"{test_c_index:.4f}",
                        'Gap': f"{train_c_index - test_c_index:.4f}"}

        for time_point in evaluation_times:
            train_auc, train_ci_lower, train_ci_upper = time_dependent_auc(
                best_model, X_train, T_train, E_train, time_point
            )
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
                print(f" 训练集: {train_auc:.3f} (95%CI: {train_ci_lower:.3f}-{train_ci_upper:.3f})")
            else:
                print(f" 训练集: 无法计算")
            if not np.isnan(test_auc):
                print(f" 测试集: {test_auc:.3f} (95%CI: {test_ci_lower:.3f}-{test_ci_upper:.3f})")
            else:
                print(f" 测试集: 无法计算")

        ssvm_results_df = pd.DataFrame([ssvm_results])
        ssvm_results_path = os.path.join(output_dir, "4_ssvm_model_results.csv")
        ssvm_results_df.to_csv(ssvm_results_path, index=False)
        print(f"\nSSVM模型结果已保存至: {ssvm_results_path}")

    else:
        print("尝试使用基于转换的SSVM方法...")


        def survival_to_classification(T, E, time_point):
            y_binary = np.zeros(len(T))
            y_binary[(T <= time_point) & (E == 1)] = 1
            y_binary[T > time_point] = 0
            mask = (T <= time_point) & (E == 0)
            y_binary = np.delete(y_binary, mask)
            return y_binary


        time_point = 36
        y_train_binary = survival_to_classification(T_train, E_train, time_point)
        mask_train = ~((T_train <= time_point) & (E_train == 0))
        X_train_binary = X_train[mask_train].copy()

        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV

        param_grid_svc = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'],
            'random_state': [42]
        }

        svc = SVC(probability=True)
        grid_search_svc = GridSearchCV(
            svc,
            param_grid_svc,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        print("开始SVM分类器网格搜索...")
        grid_search_svc.fit(X_train_binary, y_train_binary)

        print(f"最佳参数: {grid_search_svc.best_params_}")
        print(f"最佳AUC: {grid_search_svc.best_score_:.4f}")

        svc_best = grid_search_svc.best_estimator_

        ssvm_model_path = os.path.join(output_dir, "4_ssvm_svc_model.pkl")
        joblib.dump(svc_best, ssvm_model_path)
        print(f"SVM分类器模型已保存至: {ssvm_model_path}")

        y_test_binary = survival_to_classification(T_test, E_test, time_point)
        mask_test = ~((T_test <= time_point) & (E_test == 0))
        X_test_binary = X_test[mask_test].copy()

        y_train_pred = svc_best.predict_proba(X_train_binary)[:, 1]
        y_test_pred = svc_best.predict_proba(X_test_binary)[:, 1]

        train_auc = roc_auc_score(y_train_binary, y_train_pred)
        test_auc = roc_auc_score(y_test_binary, y_test_pred)

        print(f"\n36个月时间点分类AUC:")
        print(f"训练集AUC: {train_auc:.4f}")
        print(f"测试集AUC: {test_auc:.4f}")

        ssvm_results = {'Model': 'Survival SVM (Binary at 36m)'}
        ssvm_results['Train_AUC_36m'] = f"{train_auc:.3f}"
        ssvm_results['Test_AUC_36m'] = f"{test_auc:.3f}"
        ssvm_results['Train_C_index'] = "N/A"
        ssvm_results['Test_C_index'] = "N/A"

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

all_results = []

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
    try:
        if hasattr(model, 'predict_partial_hazard'):
            risk_scores = model.predict_partial_hazard(X)
            if hasattr(risk_scores, 'values'):
                risk_scores = risk_scores.values
        elif hasattr(model, 'predict'):
            risk_scores = model.predict(X)
        else:
            return

        y_true_binary = np.zeros(len(X))
        y_true_binary[(T <= time_point) & (E == 1)] = 1
        y_true_binary[(T > time_point)] = 0

        valid_mask = (T > time_point) | ((T <= time_point) & (E == 1))
        y_true_binary = y_true_binary[valid_mask]
        y_score_binary = risk_scores[valid_mask]

        if len(np.unique(y_true_binary)) < 2:
            return

        fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    except Exception as e:
        print(f"绘制{model_name} ROC曲线时出错: {e}")


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

if models_to_plot:
    for time_point in evaluation_times:
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
