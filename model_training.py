import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import joblib
import os
import warnings

# 导入十大主流回归模型
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ================= 1. 基础设置与数据加载 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('pictures', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("正在加载 Q1 筛选后的 Top 15 特征数据集...")
df = pd.read_csv('with_blood_top15.csv')
X = df.drop(columns=['血糖'])
y = df['血糖']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 【核心优化】：对数变换消除右偏态，提升回归精度
y_train_log = np.log1p(y_train)

# ================= 2. 定义十个评估模型与参数空间 =================
# 涵盖 树模型(非线性)、线性回归家族、支持向量机、神经网络
models = {
    "LightGBM": lgb.LGBMRegressor(random_state=42, verbose=-1),
    "XGBoost": xgb.XGBRegressor(random_state=42, objective='reg:squarederror'),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "ExtraTrees": ExtraTreesRegressor(random_state=42, n_jobs=-1),
    "Ridge": RidgeCV(),
    "Lasso": LassoCV(random_state=42, cv=5),
    "ElasticNet": ElasticNetCV(random_state=42, cv=5),
    "SVR": SVR(),
    "MLP_NeuralNet": MLPRegressor(random_state=42, max_iter=500)
}

# 为需要复杂调参的树模型设置超参数搜索空间，线性模型通过CV自动调参
param_distributions = {
    "LightGBM": {'n_estimators': [100, 300, 500], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7, -1]},
    "XGBoost": {'n_estimators': [100, 300, 500], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]},
    "GradientBoosting": {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1]},
    "RandomForest": {'n_estimators': [100, 300], 'max_depth': [None, 10, 20]},
    "ExtraTrees": {'n_estimators': [100, 300], 'max_depth': [None, 10, 20]},
    "SVR": {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
    "MLP_NeuralNet": {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001]}
}

# ================= 3. 模型遍历与寻优训练 =================
results = []
best_overall_model = None
best_model_name = ""
best_pmse = float('inf')  # 均方误差越小越好

print("\n🚀 开始十大模型遍历训练与超参数寻优 (这可能需要几分钟)...\n")

for name, model in models.items():
    print(f"正在训练与评估: {name} ...")

    if name in param_distributions:
        # 使用随机网格搜索防过拟合，追求极致性能
        search = RandomizedSearchCV(
            estimator=model, param_distributions=param_distributions[name],
            n_iter=10, scoring='neg_mean_squared_error', cv=5, random_state=42, n_jobs=-1
        )
        search.fit(X_train, y_train_log)
        current_best_model = search.best_estimator_
    else:
        # 线性模型自带CV或无需复杂调参
        current_best_model = model.fit(X_train, y_train_log)

    # 预测并逆对数还原
    y_pred_log = current_best_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    # 核心指标计算
    pmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)

    results.append({'Model': name, 'PMSE (MSE)': pmse, 'MAE': mae, 'MAPE (%)': mape, 'R2 (Optional)': r2})

    # 记录 PMSE (预测均方误差) 最小的王者模型
    if pmse < best_pmse:
        best_pmse = pmse
        best_overall_model = current_best_model
        best_model_name = name

# ================= 4. 保存王者模型供 Q4 使用 =================
# 保存模型 A：回归器
model_path = f'models/best_glucose_regressor_{best_model_name}.pkl'
joblib.dump(best_overall_model, model_path)

# ================= 5. 结果汇总与可视化 =================
results_df = pd.DataFrame(results).sort_values(by='PMSE (MSE)', ascending=True)
print("\n================= 十大模型评估排行榜 =================\n")
print(results_df.to_markdown(index=False))
print(f"\n🏆 最终优胜的回归模型是: {best_model_name}")
print(f"✅ 模型已成功序列化保存至: {model_path} (供 Q4 预测血糖使用)")

results_df.to_csv('10models_evaluation.csv', index=False)

# --- 图 07: 前 5 名模型评估指标对比条形图（带数值标签）---
top5_df = results_df.head(5)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Top 5 回归模型核心评价指标对比', fontsize=16, y=1.05)

# 子图1：PMSE
sns.barplot(x='PMSE (MSE)', y='Model', data=top5_df, ax=axes[0], palette='flare')
axes[0].set_title('PMSE 预测均方误差 (越低越好)')
for container in axes[0].containers:
    axes[0].bar_label(container, label_type='edge', padding=2, fmt='%.2f')

# 子图2：MAE
sns.barplot(x='MAE', y='Model', data=top5_df, ax=axes[1], palette='crest')
axes[1].set_title('MAE 平均绝对误差 (越低越好)')
for container in axes[1].containers:
    axes[1].bar_label(container, label_type='edge', padding=2, fmt='%.2f')

# 子图3：MAPE
sns.barplot(x='MAPE (%)', y='Model', data=top5_df, ax=axes[2], palette='magma')
axes[2].set_title('MAPE 绝对百分比误差 (越低越好)')
for container in axes[2].containers:
    axes[2].bar_label(container, label_type='edge', padding=2, fmt='%.1f%%')

plt.tight_layout()
plt.savefig('pictures/07_top5_models_metrics.png', bbox_inches='tight', dpi=300)
plt.close()

# --- 图 08: 优胜模型的 真实值 vs 预测值 散点拟合图 ---
y_pred_best = np.expm1(best_overall_model.predict(X_test))

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_best, alpha=0.6, color='teal', edgecolors='k')
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线 (y=x)')
plt.title(f'真实血糖值 vs 预测血糖值 ({best_model_name})', fontsize=14)
plt.xlabel('真实血糖值', fontsize=12)
plt.ylabel('预测血糖值', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('pictures/08_best_model_actual_vs_pred.png', dpi=300)
plt.close()

# --- 图 09: 优胜模型的 残差分布直方图 ---
residuals = y_test - y_pred_best
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple', bins=40)
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.title(f'模型预测残差分布 ({best_model_name})', fontsize=14)
plt.xlabel('残差 (真实值 - 预测值)', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.tight_layout()
plt.savefig('pictures/09_best_model_residuals.png', dpi=300)
plt.close()

# --- 图 10: 优胜模型特征重要性图（带数值标签）---
# 提取最优模型的特征重要性 (如果该模型支持，如树模型)
if hasattr(best_overall_model, 'feature_importances_'):
    importances = best_overall_model.feature_importances_
    importances = (importances / importances.sum()) * 100

    importance_df = pd.DataFrame({'特征': X.columns, '重要性(%)': importances}).sort_values(by='重要性(%)',
                                                                                            ascending=False)

    plt.figure(figsize=(10, 8))
    ax = sns.barplot(x='重要性(%)', y='特征', data=importance_df, palette='rocket')
    plt.title(f'优胜模型核心特征重要性贡献度 ({best_model_name})', fontsize=14)
    plt.xlabel('特征重要性占比 (%)')
    plt.ylabel('')

    # 添加数值标签（显示百分比，保留一位小数）
    for container in ax.containers:
        ax.bar_label(container, label_type='edge', padding=2, fmt='%.1f%%')

    plt.tight_layout()
    plt.savefig('pictures/10_best_model_feature_importance.png', dpi=300)
    plt.close()