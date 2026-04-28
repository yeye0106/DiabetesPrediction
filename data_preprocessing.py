import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. 环境与配置初始化
# ==========================================
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 设置中文字体，防止图表中的中文显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei']  # Mac 用户可以改为 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False

data_dir = 'processed_data'
pic_dir = 'pictures'
os.makedirs(data_dir, exist_ok=True)
os.makedirs(pic_dir, exist_ok=True)  # 确保图片文件夹存在

# ==========================================
# 1. 数据加载与基础清洗 (规则 1 & 2)
# ==========================================
print("1. 正在加载并进行基础清洗...")
train_df = pd.read_csv('with_blood.csv')
test_df = pd.read_csv('within_blood.csv')

# 规则 1：删除id、体检日期字段
drop_cols = ['id', '体检日期']
train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], inplace=True)
test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], inplace=True)

# 针对高缺失率且低相关的乙肝五项，直接删除 (缺失率>75%)
hepa_cols = ['乙肝e抗原', '乙肝e抗体', '乙肝核心抗体', '乙肝表面抗体', '乙肝表面抗原']
train_df.drop(columns=[c for c in hepa_cols if c in train_df.columns], inplace=True)
test_df.drop(columns=[c for c in hepa_cols if c in test_df.columns], inplace=True)

# 规则 2：性别使用01编码。训练集中有1个性别缺失，用众数填充
train_df['性别'] = train_df['性别'].fillna(train_df['性别'].mode()[0]).map({'男': 1, '女': 0})
test_df['性别'] = test_df['性别'].map({'男': 1, '女': 0})

# 提取因变量血糖，不参与后续的异常值、插值和标准化处理
y_train = train_df['血糖']
train_X = train_df.drop(columns=['血糖'])
test_X = test_df.copy()

# ==========================================
# ★ 核心修复：强制对齐测试集与训练集的特征列顺序
# ==========================================
# 保证 test_X 的列名和顺序与 train_X 完全一致，防止 scikit-learn 报错
test_X = test_X[train_X.columns]

# 保存列名顺序，保证后续还原为DataFrame时列名一致
feature_columns = train_X.columns

# ==========================================
# 2. 缺失值处理：KNN 插值法 (规则 3)
# ==========================================
print("2. 正在执行 KNN 插值处理缺失值...")
# 注意：KNN Imputer 必须在训练集上 fit，然后 transform 训练集和测试集，防止数据泄露
knn_imputer = KNNImputer(n_neighbors=5)

# 拟合并转换训练集
train_X_imputed = knn_imputer.fit_transform(train_X)
train_X = pd.DataFrame(train_X_imputed, columns=feature_columns)

# 仅转换测试集 (现在特征顺序一致了，不会再报错)
test_X_imputed = knn_imputer.transform(test_X)
test_X = pd.DataFrame(test_X_imputed, columns=feature_columns)

# ==========================================
# 3. 异常值处理：3σ 原则与 1%-99% 缩尾 (规则 4)
# ==========================================
print("3. 正在执行 3σ 异常检测与分位数缩尾...")
# 遍历所有特征列（性别除外）
numeric_features = [col for col in feature_columns if col != '性别']

for col in numeric_features:
    # 严格使用训练集的均值和标准差计算 3σ 边界
    mu = train_X[col].mean()
    std = train_X[col].std()

    # 获取训练集的 1% 和 99% 分位数
    lower_bound = train_X[col].quantile(0.01)
    upper_bound = train_X[col].quantile(0.99)

    # 处理训练集异常值：如果存在 |x-μ|>3σ 的情况，进行缩尾压制
    train_outliers_mask = np.abs(train_X[col] - mu) > 3 * std
    if train_outliers_mask.any():
        train_X[col] = np.clip(train_X[col], lower_bound, upper_bound)

    # 处理测试集：同样使用训练集的统计量和分位数来缩尾，保证规则一致
    test_outliers_mask = np.abs(test_X[col] - mu) > 3 * std
    if test_outliers_mask.any():
        test_X[col] = np.clip(test_X[col], lower_bound, upper_bound)

# ==========================================
# 4. 标准化处理 (规则 5)
# ==========================================
print("4. 正在执行 Z-score 标准化...")
cols_to_scale = [col for col in feature_columns if col != '性别']

# Z = (x - μ_train) / σ_train
train_mean = train_X[cols_to_scale].mean()
train_std = train_X[cols_to_scale].std()

# 防止标准差为0导致的除以0错误
train_std = train_std.replace(0, 1e-10)

train_X[cols_to_scale] = (train_X[cols_to_scale] - train_mean) / train_std
test_X[cols_to_scale] = (test_X[cols_to_scale] - train_mean) / train_std

# ==========================================
# 5. 保存处理后的数据
# ==========================================
print("5. 正在导出预处理完毕的数据...")

# 将因变量拼回训练集
final_train_df = train_X.copy()
final_train_df['血糖'] = y_train.values

train_save_path = os.path.join(data_dir, 'train_preprocessed.csv')
test_save_path = os.path.join(data_dir, 'test_preprocessed.csv')

final_train_df.to_csv(train_save_path, index=False, encoding='utf-8-sig')
test_X.to_csv(test_save_path, index=False, encoding='utf-8-sig')

# ==========================================
# 6. 数据分布可视化与保存 (新增)
# ==========================================
print("6. 正在绘制并保存分布箱线图...")

# 6.1 绘制所有连续特征标准化后的箱线图
plt.figure(figsize=(12, max(8, len(cols_to_scale) * 0.4))) # 根据特征数量动态调整高度
sns.boxplot(data=train_X[cols_to_scale], orient='h', palette='Set2')
plt.title('连续特征清洗与标准化后的箱线图分布 (检查缩尾效果)', fontsize=15, pad=15)
plt.xlabel('Z-score (标准化值)', fontsize=12)
plt.ylabel('特征名称', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '0_features_cleaned_boxplot.png'), dpi=300)
plt.close()

# 6.2 绘制目标变量（血糖）的直方图和箱线图组合
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 直方图
sns.histplot(y_train, bins=50, kde=True, ax=axes[0], color='coral', edgecolor='black', alpha=0.7)
axes[0].set_title('目标变量 (血糖) 数据分布直方图', fontsize=14)
axes[0].set_xlabel('血糖值 (mmol/L)', fontsize=12)
axes[0].set_ylabel('频数', fontsize=12)

# 箱线图
sns.boxplot(x=y_train, ax=axes[1], color='lightgreen', width=0.4)
axes[1].set_title('目标变量 (血糖) 箱线图 (观察离群值)', fontsize=14)
axes[1].set_xlabel('血糖值 (mmol/L)', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '0_target_blood_sugar_distribution.png'), dpi=300)
plt.close()

print("-" * 50)
print("✅ 预处理严格按照5条规则完成！测试集列序已强制对齐。")
print(f"📊 数据清洗验证图表已成功保存至 {pic_dir}/ 目录。")