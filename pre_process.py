import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# ================= 0. 数据加载 =================
print("正在加载数据...")
train_df = pd.read_csv('with_blood.csv')
test_df = pd.read_csv('within_blood.csv')

# ================= 1. 删除无关字段 & 提取因变量 =================
print("正在执行：删除id、体检日期，提取血糖字段...")
cols_to_drop = ['id', '体检日期']
train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
test_df = test_df.drop(columns=cols_to_drop, errors='ignore')

# 将“血糖”作为因变量单独提取，不参与后续的特征缩放和插补
target_col = '血糖'
y_train = train_df[target_col].copy()
train_df = train_df.drop(columns=[target_col])

# ================= 2. 性别编码 =================
print("正在执行：性别 0-1 编码...")
gender_map = {'男': 1, '女': 0}
# 假设数据中性别为男/女，如果存在缺失值，map后会变成NaN，将在后续KNN中被插补
train_df['性别'] = train_df['性别'].map(gender_map)
test_df['性别'] = test_df['性别'].map(gender_map)

# ================= 3. 处理高比例缺失特征 =================
print("正在执行：剔除高缺失率特征...")
# 以训练集的缺失率为基准，超过60%（涵盖了约76%缺失的乙肝相关指标）直接删除
missing_ratios = train_df.isnull().sum() / len(train_df)
high_missing_cols = missing_ratios[missing_ratios > 0.6].index.tolist()

train_df = train_df.drop(columns=high_missing_cols)
test_df = test_df.drop(columns=high_missing_cols)

# 获取剩余的所有特征名称，确保训练集和测试集特征对齐
features = train_df.columns.tolist()
# 强制让测试集的列顺序与训练集完全一致，避免后续出现 ValueError
test_df = test_df[features]

# ================= 4. 异常值处理 (1% - 99% 分位数缩尾) =================
print("正在执行：异常值分位数缩尾...")
# 连续型变量进行缩尾，排除性别这种二分类指标
continuous_features = [f for f in features if f != '性别']

# 注意：分位数上限和下限必须由【训练集】计算得出
lower_bounds = train_df[continuous_features].quantile(0.01)
upper_bounds = train_df[continuous_features].quantile(0.99)

for col in continuous_features:
    # 使用训练集的边界去限制训练集和测试集
    train_df[col] = train_df[col].clip(lower=lower_bounds[col], upper=upper_bounds[col])
    test_df[col] = test_df[col].clip(lower=lower_bounds[col], upper=upper_bounds[col])

# ================= 5. 缺失值插补 (KNN多变量插补) =================
print("正在执行：KNN多变量插补...")
# 以训练集去 fit 模型
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
train_imputed = knn_imputer.fit_transform(train_df)
# 测试集仅做 transform，不参与 fit
test_imputed = knn_imputer.transform(test_df)

train_df = pd.DataFrame(train_imputed, columns=features)
test_df = pd.DataFrame(test_imputed, columns=features)

# ================= 6. 标准化处理 (Z-score) =================
print("正在执行：特征标准化...")
scaler = StandardScaler()

# 使用训练集的均值和方差去 fit
scaler.fit(train_df[continuous_features])
# 分别对训练集和测试集进行 transform
train_df[continuous_features] = scaler.transform(train_df[continuous_features])
test_df[continuous_features] = scaler.transform(test_df[continuous_features])

# ================= 7. 特征重新排序与合并因变量 =================
print("正在执行：特征排序整合与导出...")
# 对处理后的特征按拼音/字母排序，看起来更舒服
sorted_features = sorted(features)
train_df = train_df[sorted_features]
test_df = test_df[sorted_features]

# 将目标变量“血糖”拼接到训练集的最后一列
train_df['血糖'] = y_train.values

# ================= 8. 保存文件 =================
train_df.to_csv('with_blood_processed.csv', index=False)
test_df.to_csv('within_blood_processed.csv', index=False)
print("预处理完成！文件已保存为 'with_blood_processed.csv' 和 'within_blood_processed.csv'。")