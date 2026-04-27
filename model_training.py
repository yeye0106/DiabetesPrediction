import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 环境准备
data_dir = 'processed_data'
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# 2. 加载数据
print("正在加载预处理数据进行回归建模...")
train_df = pd.read_csv(os.path.join(data_dir, 'train_preprocessed.csv'))

# 自动获取 Top 15 特征
try:
    feature_eval = pd.read_csv(os.path.join(data_dir, 'feature_evaluation_metrics.csv'))
    top_features = feature_eval['Feature'].head(15).tolist()
except:
    top_features = [col for col in train_df.columns if col not in ['血糖', '血糖_Log1p', '血糖_原始']]

X = train_df[top_features]
y = np.log1p(train_df['血糖'])  # 对血糖进行对数平滑提升 R2

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 定义评估函数
def evaluate_model(model, model_name, X_train, y_train, X_val, y_val):
    """训练模型并返回原始尺度的 MAE, RMSE, R2"""
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_val)
    y_pred = np.expm1(y_pred_log)
    y_val_orig = np.expm1(y_val)
    mae = mean_absolute_error(y_val_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred))
    r2 = r2_score(y_val_orig, y_pred)
    return model, mae, rmse, r2

# 4. 分别训练并评估每个基模型
print("\n开始训练并评估各模型...")
results = []

# LGBM
model_names = ['LGBM', 'XGB', 'RF', 'CatBoost']
base_models = [
    LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, verbose=-1),
    XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5),
    RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=0)
]

for name, model in zip(model_names, base_models):
    print(f"  正在评估 {name} ...")
    trained_model, mae, rmse, r2 = evaluate_model(model, name, X_train, y_train, X_val, y_val)
    results.append([name, mae, rmse, r2])

# 5. 构建并评估 Stacking 集成
print("  正在评估 Stacking 集成 ...")
estimators = [
    ('lgb', LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, verbose=-1)),
    ('xgb', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)),
    ('cb', CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=0))
]
stacking_reg = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), cv=5)
stacking_reg, mae, rmse, r2 = evaluate_model(stacking_reg, 'Stacking', X_train, y_train, X_val, y_val)
results.append(['Stacking', mae, rmse, r2])

# 6. 整理为 DataFrame 并输出
metrics_df = pd.DataFrame(results, columns=['Model', 'MAE', 'RMSE', 'R2'])
metrics_df = metrics_df.sort_values('R2', ascending=False)  # 按 R2 降序

print("\n" + "="*50)
print("各模型详细评估指标（原始血糖尺度）：")
print(metrics_df.to_string(index=False))
print("="*50)

# 7. 保存 CSV
metrics_csv_path = os.path.join(data_dir, 'model_comparison_metrics.csv')
metrics_df.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
print(f"✅ 模型对比指标已保存至：{metrics_csv_path}")

# 8. 保存最终的 Stacking 模型（可根据效果更换为最佳模型）
joblib.dump(stacking_reg, os.path.join(model_dir, 'best_blood_sugar_stacking_model.pkl'))
joblib.dump(top_features, os.path.join(model_dir, 'model_features.pkl'))
print("✅ 最优模型（Stacking）及特征列表已保存。")