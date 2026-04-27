import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import warnings

# 忽略一些计算过程中的常规警告
warnings.filterwarnings('ignore')

# ==========================================
# 0. 环境与配置初始化
# ==========================================
# 设置中文字体，防止图表中的中文显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows用SimHei，Mac用户请改为 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False

pic_dir = 'pictures'
data_dir = 'processed_data'
os.makedirs(pic_dir, exist_ok=True)

# ==========================================
# 1. 加载预处理后的干净数据
# ==========================================
print("正在加载预处理数据...")
train_df = pd.read_csv(os.path.join(data_dir, 'train_preprocessed.csv'))

# 提取特征 X 和目标变量 y
y = train_df['血糖']
X = train_df.drop(columns=['血糖'])

# 初始化一个 DataFrame 用于存储所有特征的评估指标
feature_eval = pd.DataFrame({'Feature': X.columns})

# ==========================================
# 2. 特征筛选四大方法计算
# ==========================================

# (1) Pearson 相关系数 (评估单变量线性相关性)
print("正在计算 Pearson 相关系数...")
pearson_corr = X.apply(lambda col: col.corr(y))
feature_eval['Pearson_Corr'] = pearson_corr.values
feature_eval['Pearson_Abs'] = np.abs(feature_eval['Pearson_Corr'])  # 取绝对值用于最终打分

# (2) VIF 多重共线性检验 (评估特征之间的冗余度)
print("正在计算整体 VIF...")
# 预处理已经排除了 NaN，此处可以安全计算
X_vif = add_constant(X)
vif_values = [variance_inflation_factor(X_vif.values, i) for i in range(1, X_vif.shape[1])]
feature_eval['VIF'] = vif_values

# (3) Lasso 回归 (L1 正则化，自动特征选择)
print("正在运行 Lasso 回归...")
lasso = LassoCV(cv=5, random_state=42).fit(X, y)
feature_eval['Lasso_Coef'] = lasso.coef_
feature_eval['Lasso_Abs'] = np.abs(feature_eval['Lasso_Coef'])  # 取绝对值用于打分

# (4) 随机森林 (评估非线性和特征交互重要性)
print("正在运行 随机森林...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X, y)
feature_eval['RF_Importance'] = rf.feature_importances_

# ==========================================
# 3. 综合打分与排名 (权重和)
# ==========================================
print("正在进行归一化综合打分...")
# 为了将不同量纲的指标相加，使用 MinMaxScaler 将它们缩放到 0~1 区间
scaler = MinMaxScaler()

score_pearson = scaler.fit_transform(feature_eval[['Pearson_Abs']])
score_lasso = scaler.fit_transform(feature_eval[['Lasso_Abs']])
score_rf = scaler.fit_transform(feature_eval[['RF_Importance']])

# 注意：VIF 是越小越好，最优是 1，越大共线性越强。
# 因此我们对 (1 / VIF) 进行归一化，这样 VIF 越接近 1，得分越接近 1
score_vif = scaler.fit_transform(1 / feature_eval[['VIF']])

# 计算总分 (各个权重均为1，满分为 4.0)
feature_eval['Final_Score'] = score_pearson + score_vif + score_lasso + score_rf

# 按总分降序排序
feature_eval = feature_eval.sort_values(by='Final_Score', ascending=False).reset_index(drop=True)

# 提取最终优胜的 Top 15 特征
top_15_features = feature_eval['Feature'].head(15).tolist()

# 保存带有完整评估过程的 CSV
eval_save_path = os.path.join(data_dir, 'feature_evaluation_metrics.csv')
feature_eval.to_csv(eval_save_path, index=False, encoding='utf-8-sig')
print(f"✅ 特征评估各项指标已保存至: {eval_save_path}")

# ==========================================
# 4. 可视化导出 (5张指标图 + 1张热力图)
# ==========================================
print("正在绘制特征评估图表...")


def plot_barh_with_values(df, x_col, title, filename, color, ascending=False):
    """辅助画图函数：绘制水平柱状图并在柱子尾部添加数值"""
    # 提取前15个，画图时为了最好的在最上面，需要倒序排列
    plot_df = df.sort_values(by=x_col, ascending=ascending).head(15).sort_values(by=x_col, ascending=not ascending)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(plot_df['Feature'], plot_df[x_col], color=color, alpha=0.8)

    # 自动在柱状图后添加数值标签
    ax.bar_label(bars, fmt='%.4f', padding=5, color='black')

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(x_col)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, filename), dpi=300)
    plt.close()


# 4.1 绘制四大指标与最终得分柱状图
plot_barh_with_values(feature_eval, 'Pearson_Abs', 'Top 15 Pearson 相关系数绝对值', '1_Pearson_Top15.png', 'steelblue')
# VIF 越小越好，所以按升序排，取最小的 15 个画图
plot_barh_with_values(feature_eval, 'VIF', 'Top 15 最低 VIF 评分 (越接近1共线性越弱)', '2_VIF_Best15.png', 'darkorange',
                      ascending=True)
plot_barh_with_values(feature_eval, 'Lasso_Abs', 'Top 15 Lasso 权重绝对值', '3_Lasso_Top15.png', 'forestgreen')
plot_barh_with_values(feature_eval, 'RF_Importance', 'Top 15 随机森林特征重要性', '4_RF_Top15.png', 'indianred')
plot_barh_with_values(feature_eval, 'Final_Score', 'Top 15 综合特征评估总分', '5_Final_Score_Top15.png', 'purple')

# 4.2 针对最终选出的 15 个变量，绘制 Pairwise VIF (成对共线性) 热力图
# 你要求值域是 1 ~ +∞
print("正在绘制 Pairwise VIF 热力图...")
top_15_X = X[top_15_features]
corr_matrix = top_15_X.corr()

# 成对 VIF 计算公式: 1 / (1 - R^2)
# 使用 np.clip 限制最大 R^2 (如 0.999)，防止出现 1/0 导致无穷大
r_squared = np.clip(corr_matrix ** 2, a_min=0, a_max=0.999)
pairwise_vif_matrix = 1 / (1 - r_squared)
# 将对角线(自身相关)设为 NaN，以免影响热力图的色阶
np.fill_diagonal(pairwise_vif_matrix.values, np.nan)

plt.figure(figsize=(12, 10))
sns.heatmap(pairwise_vif_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
            cbar_kws={'label': 'Pairwise VIF 值 (1: 完全独立, >5: 共线性明显)'},
            vmin=1, vmax=5)  # 限制色卡最大值为 5，超过 5 的都是最深的红色，方便观察
plt.title('Top 15 优胜特征成对 VIF 热力图', fontsize=16, pad=15)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, '6_Pairwise_VIF_Heatmap.png'), dpi=300)
plt.close()

# ==========================================
# 5. 结果输出
# ==========================================
print("\n" + "=" * 50)
print("🏆 特征筛选完毕！根据权重和总分，最终胜出的 15 个黄金特征为：")
for i, feature in enumerate(top_15_features, 1):
    print(f"[{i:02d}] {feature}")
print("=" * 50)
print("\n✅ 所有 6 张图表(含具体数值)已成功保存至 pictures/ 目录。")