import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

# 1. 环境准备与配置
# 创建图片保存目录
pic_dir = 'pictures'
os.makedirs(pic_dir, exist_ok=True)

# 设置中文字体，防止图表中的中文显示为方块 (Windows 常用 SimHei，Mac 可用 Arial Unicode MS)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

print("开始加载数据...")
# 读取数据
train_df = pd.read_csv('with_blood.csv')
test_df = pd.read_csv('within_blood.csv')

# 2. 字段筛选与初步清洗
# 要删除的高缺失率与无意义字段
drop_cols = ['id', '体检日期', '乙肝e抗体', '乙肝e抗原', '乙肝核心抗体', '乙肝表面抗体', '乙肝表面抗原']

train_df.drop(columns=drop_cols, inplace=True, errors='ignore')
test_df.drop(columns=drop_cols, inplace=True, errors='ignore')

# 提取因变量并从训练集特征中剥离
y_train = train_df['血糖']
X_train = train_df.drop(columns=['血糖'])
X_test = test_df.copy() # 测试集本身没有血糖

# 3. 特征重排 (让顺序符合医学和逻辑常识，看着更舒服)
# 划分为：人口学 -> 肝功能 -> 肾功能 -> 血脂 -> 血常规
ordered_features = [
    '性别', '年龄',
    '*丙氨酸氨基转换酶', '*天门冬氨酸氨基转换酶', '*碱性磷酸酶', '*r-谷氨酰基转换酶', '*总蛋白', '白蛋白', '*球蛋白', '白球比例',
    '尿素', '肌酐', '尿酸',
    '总胆固醇', '甘油三酯', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇',
    '白细胞计数', '红细胞计数', '血红蛋白', '红细胞压积', '红细胞平均体积', '红细胞平均血红蛋白量', '红细胞平均血红蛋白浓度', '红细胞体积分布宽度',
    '血小板计数', '血小板平均体积', '血小板体积分布宽度', '血小板比积',
    '中性粒细胞%', '淋巴细胞%', '单核细胞%', '嗜酸细胞%', '嗜碱细胞%'
]
X_train = X_train[ordered_features]
X_test = X_test[ordered_features]

# 类别编码：性别 男=1, 女=0
print("正在处理性别字段...")
# 1. 强制转为字符串并去除首尾可能存在的隐形空格
X_train['性别'] = X_train['性别'].astype(str).str.strip()
X_test['性别'] = X_test['性别'].astype(str).str.strip()

# 2. 映射为 0 和 1
X_train['性别'] = X_train['性别'].map({'男': 1, '女': 0})
X_test['性别'] = X_test['性别'].map({'男': 1, '女': 0})

# 3. 兜底处理：如果依然有异常值被 map 成了 NaN，则使用训练集的“众数 (mode)”进行安全填补
train_gender_mode = X_train['性别'].mode()[0]
X_train['性别'].fillna(train_gender_mode, inplace=True)
X_test['性别'].fillna(train_gender_mode, inplace=True)

# 4. 异常值处理：利用训练集的分位数进行缩尾 (Winsorization)
print("进行异常值缩尾处理...")
# 计算训练集的 1% 和 99% 分位数 (忽略类别特征'性别')
num_cols = [col for col in X_train.columns if col != '性别']
lower_bounds = X_train[num_cols].quantile(0.01)
upper_bounds = X_train[num_cols].quantile(0.99)

# 对训练集和测试集应用缩尾 (必须使用训练集的 bound)
X_train[num_cols] = X_train[num_cols].clip(lower=lower_bounds, upper=upper_bounds, axis=1)
X_test[num_cols] = X_test[num_cols].clip(lower=lower_bounds, upper=upper_bounds, axis=1)

# 5. 可视化：绘制部分核心特征的水平箱线图 (标准化前，展示真实的物理量纲)
print("生成并保存箱线图...")
# 选取几个医学上最显著、量纲差异较大的特征进行展示，避免臃肿
plot_features = ['*丙氨酸氨基转换酶', '尿酸', '甘油三酯', '低密度脂蛋白胆固醇', '年龄', '肌酐']
plt.figure(figsize=(10, 6))
# orient='h' 保证水平绘制，字自然横向显示，极其清晰
sns.boxplot(data=X_train[plot_features], orient='h', palette='Set2')
plt.title('核心特征分布箱线图 (缩尾处理后)')
plt.xlabel('数值')
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, 'core_features_boxplot.png'), dpi=300)
plt.close()

# 6. 标准化：Z = (x - μ_train) / σ_train
print("使用训练集参数进行标准化...")
scaler = StandardScaler()
# 注意：类别特征 '性别' 一般不参与标准化，这里将其剥离，处理完再合并
X_train_num = X_train[num_cols]
X_test_num = X_test[num_cols]

# fit_transform 训练集，只 transform 测试集
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_num), columns=num_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_num), columns=num_cols, index=X_test.index)

# 7. 缺失值插补：KNN 多变量插补
print("进行 KNN 多变量高级插补...")
imputer = KNNImputer(n_neighbors=5, weights='distance')

# 在标准化后的数据上进行距离计算和插补最准
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_scaled), columns=num_cols, index=X_train.index)
X_test_imputed = pd.DataFrame(imputer.transform(X_test_scaled), columns=num_cols, index=X_test.index)

# 将未标准化的 '性别' 加回来
X_train_final = pd.concat([X_train[['性别']], X_train_imputed], axis=1)
X_test_final = pd.concat([X_test[['性别']], X_test_imputed], axis=1)

# 把因变量 '血糖' 加回训练集
train_final = pd.concat([X_train_final, y_train], axis=1)
test_final = X_test_final.copy()

# 8. 保存处理后的文件
print("保存预处理后的数据...")
train_final.to_csv('X_train_preprocessed.csv', index=False, encoding='utf-8-sig')
test_final.to_csv('X_test_preprocessed.csv', index=False, encoding='utf-8-sig')

print("阶段零预处理完成！生成文件：X_train_preprocessed.csv, X_test_preprocessed.csv")
print(f"特征重新排序为: {list(X_train_final.columns)}")