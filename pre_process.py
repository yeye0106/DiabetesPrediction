import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import os

# ================= 基础设置 =================
# 设置中文字体，防止图表中的中文显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存图片的文件夹
os.makedirs('pictures', exist_ok=True)

# 1. 加载数据
print("正在加载数据...")
train_df = pd.read_csv('with_blood.csv')
test_df = pd.read_csv('within_blood.csv')

# ================= 步骤 1 & 2: 基础清洗与编码 =================
# 删除 id 和 体检日期
cols_to_drop = ['id', '体检日期']
train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns], inplace=True)
test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], inplace=True)

# 性别编码：男->1, 女->0 (处理前后空格防止报错)
train_df['性别'] = train_df['性别'].astype(str).str.strip().map({'男': 1, '女': 0})
test_df['性别'] = test_df['性别'].astype(str).str.strip().map({'男': 1, '女': 0})

# 分离目标变量 (血糖)，防止它参与特征的填补和标准化
y_train = train_df['血糖']
X_train = train_df.drop(columns=['血糖'])
X_test = test_df.copy()

# 【关键防错】强制对齐训练集和预测集的列顺序，防止后期 fit/transform 报错
X_test = X_test[X_train.columns]

# ================= 步骤 3: 缺失值处理 =================
print("正在处理缺失值...")
# 计算训练集的缺失率，删除缺失率 > 50% 的高比例特征 (如乙肝系列)
missing_ratios = X_train.isnull().mean()
high_missing_cols = missing_ratios[missing_ratios > 0.5].index.tolist()
print(f"删除高缺失率特征: {high_missing_cols}")

X_train.drop(columns=high_missing_cols, inplace=True)
X_test.drop(columns=high_missing_cols, inplace=True)

# KNN 多变量插补 (基于训练集 fit)
imputer = KNNImputer(n_neighbors=5)
X_train_cols = X_train.columns

X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train_cols)
# 严格使用训练集的分布来填补测试集
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_train_cols)

# ================= 步骤 4: 箱线图与异常值缩尾 =================
print("正在绘制箱线图并处理异常值...")
# 选取部分显著且含义明确的生理特征输出箱线图，避免全部输出导致臃肿
plot_cols = ['尿素', '肌酐', '尿酸', '总胆固醇', '甘油三酯',
             '白细胞计数', '*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶']
plot_cols = [c for c in plot_cols if c in X_train_imputed.columns]

plt.figure(figsize=(10, 8))
# orient='h' 让箱线图横向显示，特征名称自然就是横着的，方便阅读
sns.boxplot(data=X_train_imputed[plot_cols], orient='h', palette="Set2")
plt.title('显著生理特征箱线图 (异常值处理前)', fontsize=14)
plt.xlabel('数值')
plt.ylabel('特征名称')
plt.tight_layout()
plt.savefig('pictures/feature_boxplot.png', dpi=300)
plt.close()

# 异常值缩尾处理 (1% - 99%)
# 排除性别(0/1)和年龄，仅对连续型生理指标进行缩尾
exclude_clip = ['性别', '年龄']
clip_cols = [c for c in X_train_imputed.columns if c not in exclude_clip]

for col in clip_cols:
    # 严格使用训练集的 1% 和 99% 分位数
    lower_bound = X_train_imputed[col].quantile(0.01)
    upper_bound = X_train_imputed[col].quantile(0.99)

    X_train_imputed[col] = X_train_imputed[col].clip(lower=lower_bound, upper=upper_bound)
    X_test_imputed[col] = X_test_imputed[col].clip(lower=lower_bound, upper=upper_bound)

# ================= 步骤 5: 标准化 =================
print("正在进行Z-score标准化...")
scaler = StandardScaler()

# 对除了 '性别' 以外的特征进行标准化 (年龄作为连续变量可以标准化，有助于模型收敛)
scale_cols = [c for c in X_train_imputed.columns if c != '性别']

X_train_scaled = X_train_imputed.copy()
X_test_scaled = X_test_imputed.copy()

# 严格使用训练集的 均值(μ) 和 标准差(σ) 来 transform 预测集
X_train_scaled[scale_cols] = scaler.fit_transform(X_train_imputed[scale_cols])
X_test_scaled[scale_cols] = scaler.transform(X_test_imputed[scale_cols])

# ================= 步骤 6: 特征重新排序 =================
# 让特征看起来更舒服：将 性别、年龄 放在最前，其余特征按名称拼音/字母排序
print("正在重新排序特征...")
sorted_features = sorted([c for c in X_train_scaled.columns if c not in ['性别', '年龄']])
final_cols = ['性别', '年龄'] + sorted_features

X_train_final = X_train_scaled[final_cols]
X_test_final = X_test_scaled[final_cols]

# 将目标变量 (血糖) 拼接到训练集的最后一列
train_final = X_train_final.copy()
train_final['血糖'] = y_train.values

# ================= 保存处理后的数据 =================
print("处理完成，正在保存数据...")
train_final.to_csv('with_blood_processed.csv', index=False, encoding='utf-8-sig')
X_test_final.to_csv('within_blood_processed.csv', index=False, encoding='utf-8-sig')

print("全部运行成功！处理后的文件已生成，箱线图已保存至 pictures 文件夹。")