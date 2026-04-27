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

# ================= 4. 异常值缩尾（修正为1%~99%） =================
exclude_winsor = ['性别', '年龄']
winsor_cols = [c for c in X_train_imp.columns if c not in exclude_winsor]
print("📉 正在进行 1%~99% 分位数缩尾...")
for col in winsor_cols:
    lower, upper = X_train_imp[col].quantile([0.01, 0.99])
    if lower < upper:
        X_train_imp[col] = X_train_imp[col].clip(lower, upper)
        X_test_imp[col] = X_test_imp[col].clip(lower, upper)

# ================= 5. 标准化 =================
scale_cols = [c for c in X_train_imp.columns if c not in ['性别']]
scaler = StandardScaler()
X_train_imp[scale_cols] = scaler.fit_transform(X_train_imp[scale_cols])
X_test_imp[scale_cols] = scaler.transform(X_test_imp[scale_cols])

# ================= 6. 低相关性特征筛选（匹配文字描述） =================
# 计算与血糖的皮尔逊相关系数
corr_with_target = pd.DataFrame({'feature': X_train_imp.columns, 'corr': X_train_imp.corrwith(y_train).abs()})
# 保留相关性 > 0.05 的特征，强制保留年龄/性别
keep_mask = (corr_with_target['corr'] > 0.05) | (corr_with_target['feature'].isin(['年龄', '性别']))
selected_features = corr_with_target[keep_mask]['feature'].tolist()
print(f"🎯 保留相关性特征: {len(selected_features)} 个")

X_train_final = X_train_imp[selected_features]
X_test_final = X_test_imp[selected_features]

# 重排序：性别、年龄在前，其余字母序
final_order = ['性别', '年龄'] + sorted([c for c in selected_features if c not in ['性别', '年龄']])
X_train_final = X_train_final[final_order]
X_test_final = X_test_final[final_order]

# ================= 保存结果 =================
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