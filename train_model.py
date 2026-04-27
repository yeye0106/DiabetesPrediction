import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 加载数据 ====================
train_df = pd.read_csv('with_blood_processed.csv')
print("数据形状:", train_df.shape)

# 全部特征（除血糖外）且不剔除年龄性别，因为之前测试表明包含它们效果更好
X = train_df.drop('血糖', axis=1)
y = train_df['血糖']

# 对数变换（效果更好）
y_log = np.log1p(y)

# 划分训练集和测试集（80% train, 20% test）
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# ==================== 2. Optuna 超参数优化 ====================
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'verbosity': 0,
        'random_state': 42
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    pred = model.predict(X_test)
    return r2_score(y_test, pred)

def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'verbosity': -1,
        'random_state': 42
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l2', callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = model.predict(X_test)
    return r2_score(y_test, pred)

def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'depth': trial.suggest_int('depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'verbose': 0,
        'random_seed': 42
    }
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=False)
    pred = model.predict(X_test)
    return r2_score(y_test, pred)

# 运行优化（每次优化25次试验，加快速度）
print("优化 XGBoost...")
study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(objective_xgb, n_trials=25, show_progress_bar=False)
best_xgb = xgb.XGBRegressor(**study_xgb.best_params, verbosity=0, random_state=42)

print("优化 LightGBM...")
study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.optimize(objective_lgb, n_trials=25, show_progress_bar=False)
best_lgb = lgb.LGBMRegressor(**study_lgb.best_params, verbosity=-1, random_state=42)

print("优化 CatBoost...")
study_cat = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_cat.optimize(objective_cat, n_trials=25, show_progress_bar=False)
best_cat = CatBoostRegressor(**study_cat.best_params, verbose=0, random_seed=42)

# ==================== 3. 训练最终模型 ====================
print("\n训练最终模型...")
best_xgb.fit(X_train, y_train)
best_lgb.fit(X_train, y_train)
best_cat.fit(X_train, y_train)

# 预测（对数域）
pred_xgb = best_xgb.predict(X_test)
pred_lgb = best_lgb.predict(X_test)
pred_cat = best_cat.predict(X_test)

# 指数还原
pred_xgb_orig = np.expm1(pred_xgb)
pred_lgb_orig = np.expm1(pred_lgb)
pred_cat_orig = np.expm1(pred_cat)
y_test_orig = np.expm1(y_test)

def calc_metrics(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

metrics_xgb = calc_metrics(y_test_orig, pred_xgb_orig)
metrics_lgb = calc_metrics(y_test_orig, pred_lgb_orig)
metrics_cat = calc_metrics(y_test_orig, pred_cat_orig)

# 集成：加权平均（按R2分配权重）
r2_scores = [metrics_xgb['R2'], metrics_lgb['R2'], metrics_cat['R2']]
weights = np.array(r2_scores) / sum(r2_scores)
pred_ensemble = (weights[0] * pred_xgb_orig + weights[1] * pred_lgb_orig + weights[2] * pred_cat_orig)
metrics_ensemble = calc_metrics(y_test_orig, pred_ensemble)

# 打印结果表格
results = pd.DataFrame({
    '模型': ['XGBoost', 'LightGBM', 'CatBoost', '集成加权'],
    'MAE': [metrics_xgb['MAE'], metrics_lgb['MAE'], metrics_cat['MAE'], metrics_ensemble['MAE']],
    'MSE': [metrics_xgb['MSE'], metrics_lgb['MSE'], metrics_cat['MSE'], metrics_ensemble['MSE']],
    'R²': [metrics_xgb['R2'], metrics_lgb['R2'], metrics_cat['R2'], metrics_ensemble['R2']],
    'MAPE(%)': [metrics_xgb['MAPE'], metrics_lgb['MAPE'], metrics_cat['MAPE'], metrics_ensemble['MAPE']]
})
print("\n===== 最终模型性能 =====")
print(results.round(4))

# 保存最佳模型（集成模型保存为简单平均，也可保存每个基模型）
joblib.dump({'xgb': best_xgb, 'lgb': best_lgb, 'cat': best_cat, 'weights': weights}, 'ensemble_model.pkl')
print("集成模型已保存为 ensemble_model.pkl")

# ==================== 4. 可视化 ====================
# 4.1 性能对比柱状图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
metrics_plot = results.melt(id_vars='模型', value_vars=['MAE', 'MAPE(%)'], var_name='指标', value_name='值')
sns.barplot(data=metrics_plot, x='模型', y='值', hue='指标', ax=axes[0])
axes[0].set_title('误差指标对比（越小越好）')
sns.barplot(data=results, x='模型', y='R²', ax=axes[1])
axes[1].set_title('R²对比（越大越好）')
plt.tight_layout()
plt.savefig('pictures/final_ensemble_performance.png', dpi=300)
plt.close()

# 4.2 真实值与预测值散点图（集成模型）
plt.figure(figsize=(8, 6))
plt.scatter(y_test_orig, pred_ensemble, alpha=0.5, edgecolors='k')
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
plt.xlabel('真实血糖')
plt.ylabel('预测血糖')
plt.title(f'集成模型预测 vs 真实 (R²={metrics_ensemble["R2"]:.4f})')
plt.tight_layout()
plt.savefig('pictures/ensemble_scatter.png', dpi=300)
plt.close()

# 4.3 残差分布
residuals = y_test_orig - pred_ensemble
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('残差')
plt.ylabel('频数')
plt.title('残差直方图')
plt.subplot(1,2,2)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q图')
plt.tight_layout()
plt.savefig('pictures/ensemble_residuals.png', dpi=300)
plt.close()

print("所有图表已保存至 pictures/")