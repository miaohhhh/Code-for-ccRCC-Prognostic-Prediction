import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.utils import resample
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = r""
FIG_DPI = 300
N_BOOTSTRAP = 1000

MODEL_NAMES = {
    'GBS': '梯度提升生存',
    'RSF': '随机生存森林',
    'SSVM': '生存支持向量机',
}

ROC_SETTINGS = {
    'time_points': [36, 60],
    'title_train': '训练集({time_point}月)',
    'title_test': '测试集({time_point}月)',
    'xlabel': '1-特异性',
    'ylabel': '敏感性',
    'legend_loc': 'lower right',
    'legend_template': '{model_name} (AUC = {auc}, 95% CI: {ci_lower}-{ci_upper})',
}


def format_title(template, time_point):
    time_year = int(time_point / 12)
    return template.replace('{time_year}', str(time_year)).replace('{time_point}', str(time_point))


def bootstrap_auc_ci(y_true, y_scores, n_bootstrap=1000):
    bootstrapped_scores = []
    valid_indices = np.where(~np.isnan(y_true) & ~np.isnan(y_scores))[0]

    if len(valid_indices) == 0:
        return np.nan, np.nan, np.nan

    y_true_valid = y_true[valid_indices]
    y_scores_valid = y_scores[valid_indices]

    if len(np.unique(y_true_valid)) < 2:
        return np.nan, np.nan, np.nan

    for i in range(n_bootstrap):
        indices = resample(np.arange(len(y_true_valid)), random_state=i)
        if len(np.unique(y_true_valid[indices])) < 2:
            continue
        try:
            score = roc_auc_score(y_true_valid[indices], y_scores_valid[indices])
            bootstrapped_scores.append(score)
        except:
            continue

    if not bootstrapped_scores:
        return np.nan, np.nan, np.nan

    mean_auc = np.mean(bootstrapped_scores)
    ci_lower = np.percentile(bootstrapped_scores, 2.5)
    ci_upper = np.percentile(bootstrapped_scores, 97.5)

    return mean_auc, ci_lower, ci_upper


def calculate_time_dependent_roc_with_ci(model, X, T, E, time_point, n_bootstrap=1000):
    try:
        risk_scores = model.predict(X)

        y_true = np.zeros(len(X))
        y_true[(T <= time_point) & (E == 1)] = 1

        valid_mask = ((T <= time_point) & (E == 1)) | (T > time_point)
        y_true_valid = y_true[valid_mask]
        y_score_valid = risk_scores[valid_mask]

        if len(np.unique(y_true_valid)) < 2:
            return None, None, np.nan, np.nan, np.nan

        fpr, tpr, _ = roc_curve(y_true_valid, y_score_valid)

        auc_mean, ci_lower, ci_upper = bootstrap_auc_ci(
            y_true_valid, y_score_valid, n_bootstrap
        )

        return fpr, tpr, auc_mean, ci_lower, ci_upper
    except Exception as e:
        print(f"计算ROC时出错: {e}")
        return None, None, np.nan, np.nan, np.nan


def format_legend_label(template, model_name, auc_mean, ci_lower, ci_upper):
    if np.isnan(ci_lower) or np.isnan(ci_upper):
        ci_str = "N/A"
    else:
        ci_str = f"{ci_lower:.3f}-{ci_upper:.3f}"

    label = template.replace('{model_name}', model_name)
    label = label.replace('{auc}', f"{auc_mean:.3f}")
    label = label.replace('{ci_lower}', f"{ci_lower:.3f}" if not np.isnan(ci_lower) else "N/A")
    label = label.replace('{ci_upper}', f"{ci_upper:.3f}" if not np.isnan(ci_upper) else "N/A")

    return label


def plot_roc_comparison(models_dict, X_train, T_train, E_train, X_test, T_test, E_test, config, output_dir):
    print("\n开始绘制 ROC 曲线 (计算95% CI中，请稍候)...")

    model_keys = list(models_dict.keys())
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c']

    for time_point in config['time_points']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        plt.subplots_adjust(wspace=0.35, bottom=0.25, top=0.92)

        ax1.plot([0, 1], [0, 1], color='gray', lw=2.5, linestyle='--', label='随机猜测')

        for idx, model_key in enumerate(model_keys):
            model = models_dict[model_key]
            if model is None:
                continue

            print(f" 正在计算 {model_key} (训练集, {time_point}月) 的 95% CI...")
            fpr, tpr, auc_mean, ci_lower, ci_upper = calculate_time_dependent_roc_with_ci(
                model, X_train, T_train, E_train, time_point, N_BOOTSTRAP
            )

            if fpr is not None and not np.isnan(auc_mean):
                display_name = MODEL_NAMES.get(model_key, model_key)
                label = format_legend_label(
                    config['legend_template'], display_name, auc_mean, ci_lower, ci_upper
                )
                ax1.plot(fpr, tpr, color=colors[idx], lw=3, label=label)

        title = format_title(config['title_train'], time_point)
        ax1.set_title(title, fontsize=26, fontweight='bold', pad=20)
        ax1.set_xlabel(config['xlabel'], fontsize=22, fontweight='bold', labelpad=15)
        ax1.set_ylabel(config['ylabel'], fontsize=22, fontweight='bold', labelpad=15)
        ax1.legend(loc=config['legend_loc'], fontsize=10)
        ax1.grid(True, alpha=0.3, linewidth=1)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.tick_params(labelsize=20, width=1.5, length=6)

        ax2.plot([0, 1], [0, 1], color='gray', lw=2.5, linestyle='--', label='随机猜测')

        for idx, model_key in enumerate(model_keys):
            model = models_dict[model_key]
            if model is None:
                continue

            print(f" 正在计算 {model_key} (测试集, {time_point}月) 的 95% CI...")
            fpr, tpr, auc_mean, ci_lower, ci_upper = calculate_time_dependent_roc_with_ci(
                model, X_test, T_test, E_test, time_point, N_BOOTSTRAP
            )

            if fpr is not None and not np.isnan(auc_mean):
                display_name = MODEL_NAMES.get(model_key, model_key)
                label = format_legend_label(
                    config['legend_template'], display_name, auc_mean, ci_lower, ci_upper
                )
                ax2.plot(fpr, tpr, color=colors[idx], lw=3, label=label)

        title = format_title(config['title_test'], time_point)
        ax2.set_title(title, fontsize=26, fontweight='bold', pad=20)
        ax2.set_xlabel(config['xlabel'], fontsize=22, fontweight='bold', labelpad=15)
        ax2.set_ylabel(config['ylabel'], fontsize=22, fontweight='bold', labelpad=15)
        ax2.legend(loc=config['legend_loc'], fontsize=10)
        ax2.grid(True, alpha=0.3, linewidth=1)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.tick_params(labelsize=20, width=1.5, length=6)

        save_name = f"ROC_Comparison_{time_point}m.png"
        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
        plt.close()
        print(f" ✓ 已保存: {save_name}")


def main():
    print("=" * 60)
    print("正在加载数据和模型...")
    print("=" * 60)

    try:
        train_df = pd.read_csv(os.path.join(OUTPUT_DIR, "2_train_data_scaled.csv"))
        test_df = pd.read_csv(os.path.join(OUTPUT_DIR, "2_test_data_scaled.csv"))
        features_df = pd.read_csv(os.path.join(OUTPUT_DIR, "3_Final_Features_For_Modeling.csv"))
        final_features = features_df['Final_Feature_Name'].tolist()

        X_train = train_df[final_features]
        X_test = test_df[final_features]

        T_train = train_df['DFS_time'].values
        E_train = train_df['DFS_event'].values
        T_test = test_df['DFS_time'].values
        E_test = test_df['DFS_event'].values

        print(f"✓ 数据加载成功 (Train: {len(train_df)}, Test: {len(test_df)})")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return

    models = {}

    model_files = {
        'GBS': '4_gbs_model.pkl',
        'RSF': '4_rsf_model.pkl',
        'SSVM': '4_ssvm_model.pkl'
    }

    for name, file in model_files.items():
        try:
            models[name] = joblib.load(os.path.join(OUTPUT_DIR, file))
            print(f"✓ {name} 模型加载成功")
        except:
            print(f"✗ {name} 模型加载失败，跳过")
            models[name] = None

    plot_roc_comparison(
        models_dict=models,
        X_train=X_train,
        T_train=T_train,
        E_train=E_train,
        X_test=X_test,
        T_test=T_test,
        E_test=E_test,
        config=ROC_SETTINGS,
        output_dir=OUTPUT_DIR
    )

    print("\n" + "=" * 60)
    print("所有 ROC 曲线绘制完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
