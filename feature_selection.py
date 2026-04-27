import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# 1. 环境准备与配置
pic_dir = 'pictures'
os.makedirs(pic_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

print("正在加载预处理数据...")
train_df = pd.read_csv('X_train_preprocessed.csv')
test_df = pd.read_csv('X_test_preprocessed.csv')

y_train = train_df['血糖']
X_train = train_df.drop(columns=['血糖'])
features = X_train.columns.tolist()

# 创建用于存储各项指标的 DataFrame
eval_df = pd.DataFrame(index=features)

# ==========================================
# 步骤 1：计算所有特征的原始业务指标
# ==========================================
print("正在并行计算四大特征筛选指标...")

# 1. Pearson 相关系数绝对值
eval_df['Pearson_Raw'] = X_train.corrwith(y_train).abs()

# 2. VIF 多重共线性检验
X_vif_input = add_constant(X_train)
vif_list = []
for i in range(1, X_vif_input.shape[1]):  # 跳过常数项
    vif_val = variance_inflation_factor(X_vif_input.values, i)
    vif_list.append(vif_val)
eval_df['VIF_Raw'] = vif_list

# 3. Lasso 回归系数绝对值 (使用标准化后的数据以保证系数可比)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)
eval_df['Lasso_Raw'] = np.abs(lasso.coef_)

# 4. 随机森林特征重要性
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
eval_df['RF_Raw'] = rf.feature_importances_

# ==========================================
# 步骤 2：指标归一化与综合打分 (转换为 0-100 分)
# ==========================================
# Pearson, Lasso, RF 都是越大越好：直接 MinMax 缩放
min_max = MinMaxScaler(feature_range=(0, 100))
eval_df['Pearson_Score'] = min_max.fit_transform(eval_df[['Pearson_Raw']])
eval_df['Lasso_Score'] = min_max.fit_transform(eval_df[['Lasso_Raw']])
eval_df['RF_Score'] = min_max.fit_transform(eval_df[['RF_Raw']])

# VIF 是越小越好 (1最完美，越大共线性越强)：使用倒数 1/VIF 进行缩放，使 VIF 越小得分越高
eval_df['VIF_Score'] = min_max.fit_transform(1 / eval_df[['VIF_Raw']])

# 计算综合得总分 (权重各占 25%)
eval_df['Total_Score'] = (eval_df['Pearson_Score'] +
                          eval_df['VIF_Score'] +
                          eval_df['Lasso_Score'] +
                          eval_df['RF_Score']) / 4

# 按总分降序排列
eval_df = eval_df.sort_values(by='Total_Score', ascending=False)

# 保存完整评估表为 CSV
eval_df.to_csv('Feature_Evaluation_Scores.csv', encoding='utf-8-sig')
print("已生成全特征评估打分表：Feature_Evaluation_Scores.csv")

# 获取最终优胜的 15 个核心特征
top_15_features = eval_df.head(15).index.tolist()


# ==========================================
# 步骤 3：绘图函数定制 (带数值标签的水平柱状图)
# ==========================================
def plot_bar_with_labels(data, title, xlabel, filename, unit=""):
    plt.figure(figsize=(10, 8))
    # 逆序以便让得分最高的在最上方
    data_rev = data.iloc[::-1]

    bars = plt.barh(data_rev.index, data_rev.values, color=sns.color_palette("viridis", len(data)))

    # 在每根柱子末端添加数值
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2,
                 f' {width:.3f}{unit}',
                 va='center', ha='left', fontsize=10)

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    # 调整x轴限制给文字留出空间
    plt.xlim(0, max(data.values) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, filename), dpi=300)
    plt.close()


print("正在生成可视化图表...")
top_15_df = eval_df.head(15)

# 图1-4：各单独指标的 Raw 值图 (取 Top 15 绘制，避免臃肿)
plot_bar_with_labels(top_15_df['Pearson_Raw'], 'Top 15 特征 - Pearson 相关系数绝对值', '相关系数 |r|',
                     '1_Pearson_Top15.png')
# 注意：VIF 图展示的是原始 VIF 值（越小越好）
plot_bar_with_labels(top_15_df['VIF_Raw'], 'Top 15 特征 - VIF 多重共线性值 (越低越好)', 'VIF 值', '2_VIF_Top15.png')
plot_bar_with_labels(top_15_df['Lasso_Raw'], 'Top 15 特征 - Lasso 回归惩罚系数绝对值', 'Lasso |Coef|',
                     '3_Lasso_Top15.png')
plot_bar_with_labels(top_15_df['RF_Raw'], 'Top 15 特征 - 随机森林重要度', 'Gini Importance', '4_RF_Top15.png')

# 图5：最终综合得分图
plot_bar_with_labels(top_15_df['Total_Score'], 'Top 15 特征 - 四项指标加权综合得分', '综合得分 (0-100)',
                     '5_Final_Total_Score.png', unit="分")

# ==========================================
# 步骤 4：专属 Top 15 的成对 VIF 热力图
# ==========================================
# 公式: 变量 i 与 j 的成对 VIF = 1 / (1 - R_ij^2)
top15_data = X_train[top_15_features]
corr_matrix_top15 = top15_data.corr()
# 计算成对 VIF 矩阵
pairwise_vif_matrix = 1 / (1 - corr_matrix_top15 ** 2)

# 将对角线(自身与自身，值为无穷大) 替换为 NaN，以便热力图颜色映射正常显示
np.fill_diagonal(pairwise_vif_matrix.values, np.nan)

plt.figure(figsize=(12, 10))
sns.heatmap(pairwise_vif_matrix, annot=True, fmt=".2f", cmap="Reds",
            cbar_kws={'label': '成对 VIF 值'}, vmin=1.0)
plt.title('最终 15 个优胜特征的成对 VIF 热力图 (值越大共线性越强)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '6_Top15_Pairwise_VIF_Heatmap.png'), dpi=300)
plt.close()

# ==========================================
# 步骤 5：切片保存
# ==========================================
X_train_final = X_train[top_15_features]
X_test_final = test_df[top_15_features]

pd.concat([X_train_final, y_train], axis=1).to_csv('X_train_selected.csv', index=False, encoding='utf-8-sig')
X_test_final.to_csv('X_test_selected.csv', index=False, encoding='utf-8-sig')

print("\n" + "=" * 50)
print("🏆 经过四大指标集成评估，最终优胜的 15 个字段为：")
for i, feature in enumerate(top_15_features, 1):
    score = top_15_df.loc[feature, 'Total_Score']
    print(f"{i:2d}. {feature} (综合得分: {score:.2f}分)")
print("=" * 50)
print("所有 CSV 与 6 张图片 (含 VIF 热力图) 均已生成完毕！")