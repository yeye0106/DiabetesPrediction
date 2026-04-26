import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings

# 忽略警告并设置中文字体
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows用黑体
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac用户
plt.rcParams['axes.unicode_minus'] = False


def train_glucose_prediction_models_optimized(data_path='cleaned_with_blood.csv',
                                              feature_rank_path='4_final_feature_ranking.csv'):
    print("🚀 开始执行：[高阶版] 血糖预测模型交叉验证与超参数调优...")

    pic_dir = 'picture'
    os.makedirs(pic_dir, exist_ok=True)

    # 1. 读取数据与特征提取
    df = pd.read_csv(data_path)
    if os.path.exists(feature_rank_path):
        top_features_df = pd.read_csv(feature_rank_path)
        top_15_features = top_features_df['特征名称'].head(15).tolist()
    else:
        top_15_features = [col for col in df.columns if col != '血糖']

    X = df[top_15_features]
    y = df['血糖']

    # 2. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. 数据标准化 (树模型其实不需要，但为了和Ridge保持一致且无害，继续保留)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'glucose_scaler.pkl')

    # 4. 定义模型与超参数网格 (GridSearchCV)
    # 为避免运行时间过长，这里设置了精简但核心的参数网格
    models_and_params = {
        'Ridge (岭回归)': {
            'model': Ridge(),
            'params': {'alpha': [0.1, 1.0, 10.0, 100.0, 500.0]}
        },
        'Random Forest (随机森林)': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 8],  # 限制深度，防止过拟合
                'min_samples_leaf': [2, 5]
            }
        },
        'XGBoost (极端梯度提升)': {
            'model': XGBRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5],
                'reg_lambda': [1.0, 10.0]  # L2正则化，对抗噪声
            }
        }
    }

    results = []
    best_model_name = ""
    best_test_r2 = -float('inf')
    best_model_instance = None
    y_pred_best = None

    print("\n⚔️ 5折交叉验证 & 超参数搜索进行中 (请耐心等待1-2分钟)...")

    for name, mp in models_and_params.items():
        print(f"👉 正在调优训练: {name}...")
        grid_search = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        best_estimator = grid_search.best_estimator_

        # 获取训练集指标 (关键：用来判断过拟合)
        y_train_pred = best_estimator.predict(X_train_scaled)
        train_r2 = r2_score(y_train, y_train_pred)

        # 获取测试集指标
        y_test_pred = best_estimator.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)

        results.append({
            '模型名称': name,
            '最佳参数': str(grid_search.best_params_),
            '训练集 R²': round(train_r2, 4),
            '测试集 R²': round(test_r2, 4),
            '过拟合落差': round(train_r2 - test_r2, 4),
            '测试集 RMSE': round(test_rmse, 4)
        })

        print(f"   ✔️ 最佳参数: {grid_search.best_params_}")
        print(f"   ✔️ Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | 落差: {train_r2 - test_r2:.4f}\n")

        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_model_name = name
            best_model_instance = best_estimator
            y_pred_best = y_test_pred

    # 5. 保存并打印结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_evaluation_metrics_optimized.csv', index=False, encoding='utf-8-sig')
    print("\n📊 模型综合对比结果 (带过拟合分析)：")
    print(results_df[['模型名称', '训练集 R²', '测试集 R²', '过拟合落差', '测试集 RMSE']].to_string(index=False))

    joblib.dump(best_model_instance, 'best_glucose_model.pkl')
    print(f"\n🏆 最终胜出的最佳模型是: {best_model_name}")

    # ================= 优化后的可视化 =================

    # 图1：训练集 vs 测试集 R² 对比 (直观展示过拟合)
    x_labels = results_df['模型名称'].str.split(' ').str[0]
    x = np.arange(len(x_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, results_df['训练集 R²'], width, label='训练集 $R^2$', color='lightsteelblue')
    ax.bar(x + width / 2, results_df['测试集 R²'], width, label='测试集 $R^2$', color='coral')

    ax.set_ylabel('决定系数 ($R^2$)')
    ax.set_title('图1：模型 $R^2$ 性能对比 (Train vs Test)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, '5_r2_overfitting_comparison.png'), dpi=300)
    plt.close()

    # 图2：预测值 vs 残差 散点图 (异方差性检验)
    residuals = y_test - y_pred_best
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_best, residuals, alpha=0.5, color='darkorange')
    plt.axhline(y=0, color='red', linestyle='--', lw=2)
    plt.title(f'图2：预测值 vs 残差分布图 (异方差性检验)\n(最佳模型: {best_model_name.split()[0]})', fontsize=16,
              fontweight='bold')
    plt.xlabel('预测血糖值 (y_pred)', fontsize=12)
    plt.ylabel('残差 (y_test - y_pred)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, '6_heteroscedasticity_check.png'), dpi=300)
    plt.close()

    print("\n✅ 优化版建模全部完成！")


if __name__ == "__main__":
    train_glucose_prediction_models_optimized()