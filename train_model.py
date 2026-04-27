import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
import warnings

# 1. 精准控制日志：屏蔽所有烦人的警告和框架原生打印，只保留进度
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.INFO)

# 环境准备与配置
pic_dir = 'pictures'
os.makedirs(pic_dir, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['SimHei'] # 支持中文
plt.rcParams['axes.unicode_minus'] = False

print("🚀 启动竞赛级回归模型训练与融合流水线...")
df = pd.read_csv('X_train_selected.csv')
y = df['血糖']
X = df.drop(columns=['血糖'])

# 划分训练集和验证集 (80% 训练调参, 20% 验证防数据泄露)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集样本数: {X_train.shape[0]}, 验证集样本数: {X_val.shape[0]}")

# 2. 定义评价函数 (严格按照导师要求的 MAE, PMSE, MAPE)
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred) # 即 PMSE
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'PMSE(MSE)': mse, 'MAPE(%)': mape * 100, 'R2': r2}

# 3. 核心模型 Optuna 贝叶斯超参数寻优 (3折CV防过拟合)
kf = KFold(n_splits=3, shuffle=True, random_state=42)

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True), # 引入 L1 正则化抗过拟合
        'random_state': 42, 'n_jobs': -1
    }
    model = xgb.XGBRegressor(**params)
    rmse_scores = []
    for train_idx, val_idx in kf.split(X_train):
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = model.predict(X_train.iloc[val_idx])
        rmse_scores.append(np.sqrt(mean_squared_error(y_train.iloc[val_idx], preds)))
    return np.mean(rmse_scores)

def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'random_state': 42, 'n_jobs': -1, 'verbose': -1
    }
    model = lgb.LGBMRegressor(**params)
    rmse_scores = []
    for train_idx, val_idx in kf.split(X_train):
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = model.predict(X_train.iloc[val_idx])
        rmse_scores.append(np.sqrt(mean_squared_error(y_train.iloc[val_idx], preds)))
    return np.mean(rmse_scores)

def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 800),
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_seed': 42, 'verbose': False # 关闭灾难级别的刷屏日志
    }
    model = CatBoostRegressor(**params)
    rmse_scores = []
    for train_idx, val_idx in kf.split(X_train):
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = model.predict(X_train.iloc[val_idx])
        rmse_scores.append(np.sqrt(mean_squared_error(y_train.iloc[val_idx], preds)))
    return np.mean(rmse_scores)

# 执行调参 (为了更极致的效果，增加到 30 次尝试)
print("\n[1/4] 正在优化 XGBoost...")
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=30)

print("\n[2/4] 正在优化 LightGBM...")
study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=30)

print("\n[3/4] 正在优化 CatBoost (表格之王)...")
study_cat = optuna.create_study(direction='minimize')
study_cat.optimize(objective_cat, n_trials=30)

# 4. 构建模型矩阵
xgb_best = xgb.XGBRegressor(**study_xgb.best_params, random_state=42, n_jobs=-1)
lgb_best = lgb.LGBMRegressor(**study_lgb.best_params, random_state=42, n_jobs=-1, verbose=-1)
cat_best = CatBoostRegressor(**study_cat.best_params, random_seed=42, verbose=False)
rf_default = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
et_default = ExtraTreesRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)

# 构建 Stacking 融合模型 (使用顶级的三个树模型做基石，Ridge线性模型做最后裁决)
estimators = [
    ('XGB', xgb_best),
    ('LGBM', lgb_best),
    ('CatBoost', cat_best)
]
# final_estimator 使用 Ridge，加上一层正则化，极其稳定
stacking_model = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)

models = {
    'RandomForest': rf_default,
    'ExtraTrees': et_default,
    'XGBoost': xgb_best,
    'LightGBM': lgb_best,
    'CatBoost': cat_best,
    'Stacking_Ensemble': stacking_model # 王牌
}

# 5. 训练与终极评估
print("\n[4/4] 正在训练全体模型并进行集成融合评估...")
results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    predictions[name] = y_pred
    metrics = evaluate_model(y_val, y_pred)
    metrics['Model'] = name
    results.append(metrics)

results_df = pd.DataFrame(results).set_index('Model')

# 按 MAE 从低到高排序，MAE越低越好
results_df = results_df.sort_values(by='MAE')
print("\n🏆 ====== 终极竞赛级验证集评估结果 (按MAE排序) ======")
print(results_df.round(4))
results_df.to_csv('Model_Evaluation_Metrics_Pro.csv', encoding='utf-8-sig')

best_model_name = results_df.index[0] # 排第一的就是最强
best_preds = predictions[best_model_name]
print(f"\n🥇 综合表现最强王者: {best_model_name}")

# ==========================================
# 6. 学术/竞赛级图表输出
# ==========================================
print("\n正在生成高规格可视化图表...")

# 图1：MAE与MAPE多模型对比图
fig, ax1 = plt.subplots(figsize=(12, 6))
width = 0.35
x = np.arange(len(results_df.index))

bar1 = ax1.bar(x - width/2, results_df['MAE'], width, label='MAE (越小越好)', color='#2ca02c')
ax1.set_ylabel('Mean Absolute Error (MAE)', color='black')
ax1.set_xticks(x)
ax1.set_xticklabels(results_df.index, rotation=15)

ax2 = ax1.twinx()
bar2 = ax2.bar(x + width/2, results_df['MAPE(%)'], width, label='MAPE % (越小越好)', color='#d62728')
ax2.set_ylabel('MAPE (%)', color='black')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

# 给最强模型加上"皇冠"标记
plt.text(0, results_df.iloc[0]['MAPE(%)'] + 0.5, '👑 最强', ha='center', color='red', fontweight='bold', fontsize=12)

plt.title('四大基线模型 vs 集成模型 (Stacking) 核心指标对比', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '7_Pro_Model_Comparison.png'), dpi=300)
plt.close()

# 图2：最强王者模型 散点拟合图
plt.figure(figsize=(8, 8))
plt.scatter(y_val, best_preds, alpha=0.6, color='#1f77b4', edgecolors='w')
min_val, max_val = min(y_val.min(), best_preds.min()), max(y_val.max(), best_preds.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美拟合线')
plt.title(f'最强模型 [{best_model_name}] : 真实血糖值 vs 预测血糖值', fontsize=14)
plt.xlabel('真实血糖值')
plt.ylabel('预测血糖值')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '8_Pro_Best_Model_Scatter.png'), dpi=300)
plt.close()

print("🏁 竞赛级训练完毕！拿着这些图表和模型去展示，绝对硬核。")