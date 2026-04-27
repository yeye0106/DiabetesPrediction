import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
import warnings

# ---------------- 环境与配置 ----------------
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.INFO)

pic_dir = 'pictures'
model_dir = 'models'
os.makedirs(pic_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("📂 正在加载数据并准备终极竞赛流水线...")
df = pd.read_csv('X_train_selected.csv')
y = df['血糖']
X = df.drop(columns=['血糖'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------- 定义评价函数 ----------------
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'PMSE(MSE)': mse, 'MAPE(%)': mape * 100, 'R2': r2}


# ---------------- Optuna 贝叶斯寻优 ----------------
kf = KFold(n_splits=3, shuffle=True, random_state=42)


def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 700),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 5.0, log=True),
        'random_state': 42, 'n_jobs': -1
    }
    model = xgb.XGBRegressor(**params)
    scores = []
    for t_idx, v_idx in kf.split(X_train):
        model.fit(X_train.iloc[t_idx], y_train.iloc[t_idx])
        scores.append(np.sqrt(mean_squared_error(y_train.iloc[v_idx], model.predict(X_train.iloc[v_idx]))))
    return np.mean(scores)


def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 700),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 5.0, log=True),
        'random_state': 42, 'n_jobs': -1, 'verbose': -1
    }
    model = lgb.LGBMRegressor(**params)
    scores = []
    for t_idx, v_idx in kf.split(X_train):
        model.fit(X_train.iloc[t_idx], y_train.iloc[t_idx])
        scores.append(np.sqrt(mean_squared_error(y_train.iloc[v_idx], model.predict(X_train.iloc[v_idx]))))
    return np.mean(scores)


print("\n🔍 正在进行超参数寻优...")
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=20)
study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=20)

# ---------------- 定义与训练全模型 阵列 ----------------
best_xgb = xgb.XGBRegressor(**study_xgb.best_params, random_state=42)
best_lgb = lgb.LGBMRegressor(**study_lgb.best_params, random_state=42, verbose=-1)
# CatBoost 采用默认高性能配置（寻优耗时较长，这里给出一组竞赛常用强力参数）
best_cat = CatBoostRegressor(iterations=600, depth=6, learning_rate=0.03, l2_leaf_reg=3, random_seed=42, verbose=False)

# 增加 岭回归(Ridge) 和 随机森林(RF)
ridge_model = Ridge(alpha=1.0)
rf_model = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
et_model = ExtraTreesRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)

# Stacking 集成：使用三个王者模型做基础，Ridge做元学习器
estimators = [('XGB', best_xgb), ('LGBM', best_lgb), ('CatBoost', best_cat)]
stacking_model = StackingRegressor(estimators=estimators, final_estimator=Ridge(), cv=5)

models = {
    'Ridge_Regression': ridge_model,
    'RandomForest': rf_model,
    'ExtraTrees': et_model,
    'XGBoost': best_xgb,
    'LightGBM': best_lgb,
    'CatBoost': best_cat,
    'Stacking_Ensemble': stacking_model
}

# ---------------- 统一训练、评估、保存 ----------------
print("\n🦾 正在统一训练全模型并持久化...")
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # 评价
    metrics = evaluate_model(y_val, y_pred)
    metrics['Model'] = name
    results.append(metrics)

    # 保存模型到 models/ 文件夹
    model_path = os.path.join(model_dir, f'{name}_model.joblib')
    joblib.dump(model, model_path)
    print(f"✅ 已保存: {name}")

# 输出结果表
results_df = pd.DataFrame(results).set_index('Model').sort_values(by='MAE')
print("\n🏆 ====== 终极全模型对比看板 ======")
print(results_df.round(4))
results_df.to_csv('Full_Model_Performance.csv', encoding='utf-8-sig')

# ---------------- 可视化对比图 ----------------
plt.figure(figsize=(14, 7))
sns.barplot(x=results_df.index, y=results_df['MAE'], palette='magma')
plt.title('全阵列模型 MAE 对比 (越低越好)', fontsize=15)
plt.xticks(rotation=20)
plt.ylabel('平均绝对误差 (MAE)')
for i, val in enumerate(results_df['MAE']):
    plt.text(i, val + 0.01, f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '10_Full_Model_MAE_Comparison.png'), dpi=300)

print("\n🏁 任务完成！所有模型已保存至 'models/' 文件夹，性能表见 'Full_Model_Performance.csv'。")