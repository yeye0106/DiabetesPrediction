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

print("正在加载数据...")
train_df = pd.read_csv('with_blood.csv')
test_df = pd.read_csv('within_blood.csv')

# ================= 1. 删除无关列 + 性别编码 =================
cols_to_drop = ['id', '体检日期']
train_df = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])

train_df['性别'] = train_df['性别'].astype(str).str.strip().map({'男': 1, '女': 0})
test_df['性别'] = test_df['性别'].astype(str).str.strip().map({'男': 1, '女': 0})

mode_sex = train_df['性别'].mode()[0]
train_df['性别'] = train_df['性别'].fillna(mode_sex)
test_df['性别'] = test_df['性别'].fillna(mode_sex)

y_train = train_df['血糖'].copy()
X_train = train_df.drop(columns=['血糖'])
X_test = test_df.copy()

# 安全对齐列（防止测试集列顺序不同）
common_cols = sorted(list(set(X_train.columns) & set(X_test.columns)))
X_train = X_train[common_cols]
X_test = X_test[common_cols]

# ================= 2. 删除高缺失率特征 (>50%) =================
missing_ratios = X_train.isnull().mean()
high_missing_cols = missing_ratios[missing_ratios > 0.5].index.tolist()
print(f"✅ 删除高缺失率特征 ({len(high_missing_cols)}个): {high_missing_cols}")
X_train.drop(columns=high_missing_cols, inplace=True)
X_test.drop(columns=high_missing_cols, inplace=True)

# ================= 3. KNN插补（必须在标准化前） =================
print("🔄 正在进行KNN插补 (n_neighbors=5)...")
imputer = KNNImputer(n_neighbors=5)
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# ================= 4. 异常值处理：3σ检测 + 分位数缩尾（仅对异常值） =================
exclude_winsor = ['性别', '年龄']
winsor_cols = [c for c in X_train_imp.columns if c not in exclude_winsor]
print("📉 正在进行 3σ 异常值检测 + 1%/99% 分位数缩尾（仅替换异常值）...")
for col in winsor_cols:
    mean = X_train_imp[col].mean()
    std = X_train_imp[col].std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    lower_q, upper_q = X_train_imp[col].quantile([0.01, 0.99])

    # 标记训练集中的异常值
    is_outlier = (X_train_imp[col] < lower_bound) | (X_train_imp[col] > upper_bound)
    # 替换：小于 lower_bound 的用 lower_q，大于 upper_bound 的用 upper_q
    X_train_imp.loc[is_outlier & (X_train_imp[col] < lower_bound), col] = lower_q
    X_train_imp.loc[is_outlier & (X_train_imp[col] > upper_bound), col] = upper_q

    # 测试集使用训练集计算的 3σ 边界和分位数
    is_outlier_test = (X_test_imp[col] < lower_bound) | (X_test_imp[col] > upper_bound)
    X_test_imp.loc[is_outlier_test & (X_test_imp[col] < lower_bound), col] = lower_q
    X_test_imp.loc[is_outlier_test & (X_test_imp[col] > upper_bound), col] = upper_q

# ================= 5. 标准化 =================
scale_cols = [c for c in X_train_imp.columns if c not in ['性别']]
scaler = StandardScaler()
X_train_imp[scale_cols] = scaler.fit_transform(X_train_imp[scale_cols])
X_test_imp[scale_cols] = scaler.transform(X_test_imp[scale_cols])

# ================= 6. 保存预处理结果（全部特征，留待问题1筛选） =================
# 不进行低相关性筛选，保证问题1从原始特征池出发
X_train_final = X_train_imp.copy()
X_test_final = X_test_imp.copy()

# 重排序：性别、年龄在前，其余字母序
final_order = ['性别', '年龄'] + sorted([c for c in X_train_final.columns if c not in ['性别', '年龄']])
X_train_final = X_train_final[final_order]
X_test_final = X_test_final[final_order]

train_final = X_train_final.copy()
train_final['血糖'] = y_train.values
train_final.to_csv('with_blood_processed.csv', index=False, encoding='utf-8-sig')
X_test_final.to_csv('within_blood_processed.csv', index=False, encoding='utf-8-sig')

# ================= 可视化对比 =================
plot_cols = [c for c in final_order if c not in ['性别', '年龄']][:6]
if plot_cols:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(plot_cols):
        orig = X_train[col].dropna()
        proc = X_train_final[col]
        axes[i].boxplot([orig, proc], tick_labels=['原始', '处理后'], showfliers=False)
        axes[i].set_title(col, fontsize=10)
        axes[i].grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('pictures/before_after_boxplot.png', dpi=300)
    plt.close()
    print("📊 箱线对比图已保存至 pictures/before_after_boxplot.png")

print("✨ 全部预处理完成！已生成 with_blood_processed.csv 和 within_blood_processed.csv")