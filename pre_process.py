import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import os

# ================= 基础设置 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('pictures', exist_ok=True)

# 1. 加载数据
print("正在加载数据...")
train_df = pd.read_csv('with_blood.csv')
test_df = pd.read_csv('within_blood.csv')

# ================= 步骤 1: 删除无关列 + 性别编码（改进版） =================
cols_to_drop = ['id', '体检日期']
train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns], inplace=True)
test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], inplace=True)

# 性别清洗：先去除首尾空格，再统一替换
train_df['性别'] = train_df['性别'].astype(str).str.strip()
test_df['性别'] = test_df['性别'].astype(str).str.strip()
# 映射：男->1, 女->0，其余无法映射的变成 NaN
train_df['性别'] = train_df['性别'].map({'男': 1, '女': 0})
test_df['性别'] = test_df['性别'].map({'男': 1, '女': 0})

# 如果还有 NaN，用训练集的众数填充（此时训练集众数一定存在）
if train_df['性别'].isna().any() or test_df['性别'].isna().any():
    mode_sex = train_df['性别'].mode()[0]
    train_df['性别'] = train_df['性别'].fillna(mode_sex)
    test_df['性别'] = test_df['性别'].fillna(mode_sex)
    print(f"使用众数 {mode_sex} 填充了缺失的性别值")

# 分离目标变量
y_train = train_df['血糖']
X_train = train_df.drop(columns=['血糖'])
X_test = test_df.copy()
X_test = X_test[X_train.columns]          # 对齐列顺序

# ================= 步骤 2: 删除高缺失率特征（>50%） =================
missing_ratios = X_train.isnull().mean()
high_missing_cols = missing_ratios[missing_ratios > 0.5].index.tolist()
print(f"删除高缺失率特征: {high_missing_cols}")
X_train.drop(columns=high_missing_cols, inplace=True)
X_test.drop(columns=high_missing_cols, inplace=True)

# ================= 步骤 3: 异常值缩尾（排除性别和年龄） =================
exclude_winsor = ['性别', '年龄']
winsor_cols = [c for c in X_train.columns if c not in exclude_winsor]

for col in winsor_cols:
    lower = X_train[col].quantile(0.005)
    upper = X_train[col].quantile(0.995)
    if lower == upper:
        continue
    X_train[col] = X_train[col].clip(lower, upper)
    X_test[col] = X_test[col].clip(lower, upper)

# ================= 步骤 4: 标准化 =================
scale_cols = [c for c in X_train.columns if c != '性别']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

# ================= 步骤 5: KNN插补 =================
print("正在进行KNN插补...")
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_scaled), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test_scaled), columns=X_test.columns)

# ================= 步骤 6: 特征重排序 =================
sorted_features = sorted([c for c in X_train_imputed.columns if c not in ['性别', '年龄']])
final_cols = ['性别', '年龄'] + sorted_features
X_train_final = X_train_imputed[final_cols]
X_test_final = X_test_imputed[final_cols]

train_final = X_train_final.copy()
train_final['血糖'] = y_train.values

# ================= 保存 =================
train_final.to_csv('with_blood_processed.csv', index=False, encoding='utf-8-sig')
X_test_final.to_csv('within_blood_processed.csv', index=False, encoding='utf-8-sig')

# ================= 可视化对比（使用 tick_labels 避免警告） =================
compare_cols = [c for c in final_cols if c not in ['性别', '年龄']][:6]
if compare_cols:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(compare_cols):
        orig_data = X_train[col].dropna()
        proc_data = train_final[col]
        axes[i].boxplot([orig_data, proc_data], tick_labels=['原始', '处理后'], showfliers=False)
        axes[i].set_title(col)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('pictures/before_after_boxplot.png', dpi=300)
    plt.close()
    print("箱线对比图已保存至 pictures/before_after_boxplot.png")

print("全部处理完成！文件已保存。")