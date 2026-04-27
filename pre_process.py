import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import os

# 创建图片保存文件夹
os.makedirs('pictures', exist_ok=True)

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# ==================== 1. 读取数据 ====================
train_df = pd.read_csv('with_blood.csv')
test_df = pd.read_csv('within_blood.csv')

print("训练集形状:", train_df.shape)
print("测试集形状:", test_df.shape)

# ==================== 2. 删除无用列及高缺失列 ====================
# 删除 id 和体检日期
train_df.drop(['id', '体检日期'], axis=1, inplace=True)
test_df.drop(['id', '体检日期'], axis=1, inplace=True)

# 删除乙肝五项（缺失率>65%）
hep_cols = [col for col in train_df.columns if '乙肝' in col]
train_df.drop(hep_cols, axis=1, inplace=True)
test_df.drop(hep_cols, axis=1, inplace=True)

print("删除乙肝五项后训练集特征:", train_df.columns.tolist())
print("删除乙肝五项后测试集特征:", test_df.columns.tolist())

# ==================== 3. 性别编码 ====================
train_df['性别'] = train_df['性别'].map({'男': 0, '女': 1})
test_df['性别'] = test_df['性别'].map({'男': 0, '女': 1})

# ==================== 4. 分离特征与标签 ====================
# 训练集：血糖为目标变量 y，其余为特征 X
y_train = train_df['血糖'].copy()
X_train = train_df.drop('血糖', axis=1).copy()
# 测试集：无血糖，全部为特征 X_test
X_test = test_df.copy()

# 确保两个数据集的列完全一致（测试集可能顺序不同，但列应相同）
# 按列名排序统一
common_cols = sorted(X_train.columns.intersection(X_test.columns))
X_train = X_train[common_cols]
X_test = X_test[common_cols]
print("共同特征数:", len(common_cols))

# ==================== 5. KNN 插补缺失值 ====================
# 使用训练集拟合 KNN 插补器（n_neighbors=5, weights='distance'）
imputer = KNNImputer(n_neighbors=5, weights='distance')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 转回 DataFrame，保持列名
X_train = pd.DataFrame(X_train_imputed, columns=common_cols)
X_test = pd.DataFrame(X_test_imputed, columns=common_cols)

print("KNN 插补完成，训练集缺失值数量:", X_train.isnull().sum().sum())
print("KNN 插补完成，测试集缺失值数量:", X_test.isnull().sum().sum())

# ==================== 6. 异常值缩尾（基于训练集分位数） ====================
def winsorize_by_percentile(train_series, test_series, lower=0.01, upper=0.99):
    """根据训练集的分位数对训练集和测试集进行缩尾"""
    low = train_series.quantile(lower)
    high = train_series.quantile(upper)
    train_clipped = train_series.clip(low, high)
    test_clipped = test_series.clip(low, high)
    return train_clipped, test_clipped

# 对每个数值特征进行缩尾（所有列都是数值特征，除性别外）
for col in common_cols:
    if col != '性别':   # 性别不需要缩尾
        X_train[col], X_test[col] = winsorize_by_percentile(X_train[col], X_test[col])

print("异常值缩尾完成")

# ==================== 7. 标准化（基于训练集均值和标准差） ====================
# 注意：性别不需要标准化，年龄也先标准化（后续可根据需要选择是否使用）
scaler = StandardScaler()
# 选择需要标准化的列（所有列均进行标准化，但为保持一致性，对数值列标准化，性别可标可不标，这里统一标准化）
# 为了保持年龄作为分层变量的可比性，我们也标准化年龄
cols_to_scale = [col for col in common_cols if col != '性别']  # 性别不标准化
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

if cols_to_scale:
    scaler.fit(X_train[cols_to_scale])
    X_train_scaled[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

print("标准化完成，训练集均值:\n", X_train_scaled[cols_to_scale].mean())
print("标准化完成，训练集标准差:\n", X_train_scaled[cols_to_scale].std())

# 重新组装训练集（包含血糖 y）
train_processed = pd.concat([X_train_scaled, y_train], axis=1)
test_processed = X_test_scaled

# ==================== 8. 整理列顺序（按业务逻辑分组排序） ====================
# 定义业务分组顺序（根据常见的体检指标）
def sort_columns_by_group(df):
    # 自定义分组顺序
    group_order = {
        '基本信息': ['性别', '年龄'],
        '肝功能': [col for col in df.columns if '谷氨酰基转换酶' in col or '丙氨酸氨基转换酶' in col or '天门冬氨酸氨基转换酶' in col or '碱性磷酸酶' in col],
        '蛋白质代谢': [col for col in df.columns if '总蛋白' in col or '白蛋白' in col or '球蛋白' in col or '白球比例' in col],
        '肾功能': [col for col in df.columns if '尿素' in col or '肌酐' in col or '尿酸' in col],
        '血脂': [col for col in df.columns if '胆固醇' in col or '甘油三酯' in col],
        '血常规': [col for col in df.columns if '红细胞' in col or '白细胞' in col or '血小板' in col or '血红蛋白' in col or '中性粒细胞' in col or '淋巴细胞' in col or '单核细胞' in col or '嗜酸细胞' in col or '嗜碱细胞' in col],
        '其他': []  # 未归类的放最后
    }
    # 先按组排序，组内按原始名称排序
    ordered_cols = []
    for group, keywords in group_order.items():
        group_cols = []
        for kw in keywords:
            group_cols.extend([c for c in df.columns if kw in c and c not in ordered_cols])
        group_cols = sorted(set(group_cols))
        ordered_cols.extend(group_cols)
    # 添加未被归类的列（按名称排序）
    remaining = [c for c in df.columns if c not in ordered_cols]
    ordered_cols.extend(sorted(remaining))
    return df[ordered_cols]

train_processed = sort_columns_by_group(train_processed)
test_processed = sort_columns_by_group(test_processed)

# ==================== 9. 保存处理后的 CSV ====================
train_processed.to_csv('with_blood_processed.csv', index=False)
test_processed.to_csv('within_blood_processed.csv', index=False)
print("处理后的数据已保存: with_blood_processed.csv, within_blood_processed.csv")

# ==================== 10. 生成箱线图 ====================
# 选取关键特征（与血糖生理相关的主要指标）
key_features = [col for col in train_processed.columns if any(kw in col for kw in
                ['谷氨酰基转换酶', '丙氨酸氨基转换酶', '天门冬氨酸氨基转换酶',
                 '总胆固醇', '甘油三酯', '高密度脂蛋白', '低密度脂蛋白',
                 '尿素', '肌酐', '尿酸', '血糖'])] + ['年龄']
# 确保特征存在
key_features = [f for f in key_features if f in train_processed.columns]

# 绘制箱线图（使用处理后的数据）
plt.figure(figsize=(14, 8))
for i, col in enumerate(key_features, 1):
    plt.subplot(3, 4, i)
    data_to_plot = train_processed[col].dropna()
    sns.boxplot(y=data_to_plot, color='skyblue')
    plt.title(col)
    plt.ylabel('')
    plt.xlabel(col, rotation=0)
    plt.tight_layout()
plt.suptitle('关键特征箱线图（处理后的训练集）', fontsize=16, y=1.02)
plt.savefig('pictures/key_features_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 另外绘制所有特征的箱线图（分多张图，避免拥挤）
all_features = [c for c in train_processed.columns if c != '血糖']
n_features = len(all_features)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols
plt.figure(figsize=(16, 4*n_rows))
for i, col in enumerate(all_features, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(y=train_processed[col].dropna(), color='lightgreen')
    plt.title(col, fontsize=8)
    plt.ylabel('')
    plt.xlabel(col, rotation=60, fontsize=6)
    plt.tight_layout()
plt.suptitle('全特征箱线图（训练集处理完）', fontsize=16, y=1.02)
plt.savefig('pictures/all_features_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

print("箱线图已保存至 pictures/ 文件夹")

# 可选：输出数据基本信息报告
print("\n=== 数据处理报告 ===")
print(f"训练集最终形状: {train_processed.shape}")
print(f"测试集最终形状: {test_processed.shape}")
print("训练集血糖分布:")
print(train_processed['血糖'].describe())
print("\n训练集各风险区间人数:")
print(pd.cut(train_processed['血糖'], bins=[-np.inf, 6.1, 6.7, np.inf], labels=['低风险', '中风险', '高风险']).value_counts())