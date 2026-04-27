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
y = np.log1p(train_df['血糖']) # 对血糖进行对数平滑提升 R2

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 构建 Stacking 强力回归模型
print("正在优化并训练 Stacking 回归集成模型...")
base_models = [
    ('lgb', LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, verbose=-1)),
    ('xgb', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)),
    ('cb', CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=0))
]

# 元模型使用 RidgeCV 自动分配权重
stacking_reg = StackingRegressor(estimators=base_models, final_estimator=RidgeCV(), cv=5)
stacking_reg.fit(X_train, y_train)

# 4. 评估结果
y_pred_log = stacking_reg.predict(X_val)
y_pred_raw = np.expm1(y_pred_log) # 还原真实血糖
y_val_raw = np.expm1(y_val)

mae = mean_absolute_error(y_val_raw, y_pred_raw)
rmse = np.sqrt(mean_squared_error(y_val_raw, y_pred_raw))
r2 = r2_score(y_val_raw, y_pred_raw)

print("\n" + "="*30)
print(f"回归模型评估结果:\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}")
print("="*30)

# 5. 保存模型
joblib.dump(stacking_reg, os.path.join(model_dir, 'best_blood_sugar_stacking_model.pkl'))
joblib.dump(top_features, os.path.join(model_dir, 'model_features.pkl'))
print("✅ 回归模型及特征列表已保存。")