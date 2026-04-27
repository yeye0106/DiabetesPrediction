import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import warnings

warnings.filterwarnings('ignore')

# ================= 基础设置 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('pictures', exist_ok=True)

# 1. 加载预处理后的数据
print("正在加载预处理后的数据...")
df = pd.read_csv('with_blood_processed.csv')

y = df['血糖']
X = df.drop(columns=['血糖'])
features = X.columns.tolist()

# 结果收集容器
results_df = pd.DataFrame({'特征': features}).set_index('特征')


# 辅助函数：绘制带数据标签的水平条形图
def plot_bar_with_labels(series, title, filename, unit='', top_n=20, color='royalblue', ascending=False):
    plt.figure(figsize=(10, 8))
    # 排序并截取前 top_n
    data_to_plot = series.sort_values(ascending=ascending).head(top_n)

    # 绘制条形图
    ax = sns.barplot(x=data_to_plot.values, y=data_to_plot.index, palette=color)

    # 添加数据标签
    for container in ax.containers:
        ax.bar_label(container, fmt=f'%.4f {unit}', padding=5, fontsize=10)

    plt.title(title, fontsize=15, pad=15)
    plt.xlabel('数值 / 分数')
    plt.ylabel('特征名称')
    # 扩大x轴范围，防止标签被图表边缘截断
    max_val = data_to_plot.max()
    plt.xlim(0 if data_to_plot.min() >= 0 else data_to_plot.min() * 1.1, max_val * 1.2)
    plt.tight_layout()
    plt.savefig(f'pictures/{filename}', dpi=300)
    plt.close()


# ================= 第一步：Pearson 相关系数 =================
print("正在计算 Pearson 相关系数...")
corr_matrix = df.corr()
pearson_raw = corr_matrix['血糖'].drop('血糖').abs()
results_df['Pearson(绝对值)'] = pearson_raw

plot_bar_with_labels(pearson_raw, 'Top 20 Pearson相关系数 (绝对值)', '01_Pearson_Raw.png', color='Blues_r')

# ================= 第二步：VIF 共线性计算 =================
print("正在计算 VIF 多重共线性...")
# 利用相关系数矩阵的逆矩阵对角线直接计算全局 VIF，比 statsmodels 循环计算更稳健且快速
inv_corr = np.linalg.pinv(X.corr().values)
vif_raw = pd.Series(np.diag(inv_corr), index=X.columns)
# 将极小的负数或由于数值精度引起的异常值修剪为 1 (完美独立)
vif_raw = vif_raw.apply(lambda x: max(1.0, x))
results_df['VIF值'] = vif_raw

# VIF越小越好，所以这里绘图时升序排列，展示VIF最小(最独立)的特征
plot_bar_with_labels(vif_raw, 'Top 20 独立特征 (VIF值最小)', '02_VIF_Raw.png', ascending=True, color='Greens_r')

# ================= 第三步：Lasso 回归 =================
print("正在计算 Lasso 惩罚系数...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
lasso_raw = pd.Series(np.abs(lasso.coef_), index=X.columns)
results_df['Lasso(绝对系数)'] = lasso_raw

plot_bar_with_labels(lasso_raw, 'Top 20 Lasso回归系数 (绝对值)', '03_Lasso_Raw.png', color='Oranges_r')

# ================= 第四步：随机森林重要性 =================
print("正在计算 Random Forest 特征重要性...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)
rf_raw = pd.Series(rf.feature_importances_, index=X.columns)
results_df['RF(重要度)'] = rf_raw

plot_bar_with_labels(rf_raw, 'Top 20 随机森林特征重要性', '04_RF_Raw.png', color='Purples_r')

# ================= 第五步：综合打分与排名 =================
print("正在进行综合归一化打分...")
minmax = MinMaxScaler(feature_range=(0, 100))

# 1. Pearson 得分 (直接归一化)
results_df['Pearson_Score'] = minmax.fit_transform(results_df[['Pearson(绝对值)']])

# 2. VIF 得分 (取倒数后归一化：VIF越小，倒数越大，得分越高)
results_df['VIF_Score'] = minmax.fit_transform(1 / results_df[['VIF值']])

# 3. Lasso 得分 (直接归一化)
results_df['Lasso_Score'] = minmax.fit_transform(results_df[['Lasso(绝对系数)']])

# 4. RF 得分 (直接归一化)
results_df['RF_Score'] = minmax.fit_transform(results_df[['RF(重要度)']])

# 计算权重和的总分 (当前采用等权重 1:1:1:1，满分400)
results_df['综合总分'] = results_df['Pearson_Score'] + results_df['VIF_Score'] + results_df['Lasso_Score'] + results_df[
    'RF_Score']

# 按总分降序排列
results_df = results_df.sort_values(by='综合总分', ascending=False)

# 绘制总分图
plot_bar_with_labels(results_df['综合总分'], '特征综合评估总分排名 (Top 20)', '05_Final_Total_Score.png', unit='分',
                     color='magma')

# ================= 第六步：Top 15 成对 VIF 热力图 =================
print("正在绘制 Top 15 成对 VIF 热力图...")
top15_features = results_df.head(15).index.tolist()
X_top15 = X[top15_features]

# 计算成对特征的相关系数矩阵 R
R_matrix = X_top15.corr().values
# 成对 VIF 公式: 1 / (1 - R^2)
# 加上 1e-5 防止对角线 (R=1) 发生除零错误
pairwise_vif = 1 / (1 - R_matrix ** 2 + 1e-5)
pairwise_vif_df = pd.DataFrame(pairwise_vif, index=top15_features, columns=top15_features)

# 为了防止对角线的超大数值冲淡了其他区块的颜色，将热力图最大显示上限设为 10 (医学统计中通常VIF>10即视为严重共线性)
plt.figure(figsize=(12, 10))
sns.heatmap(pairwise_vif_df, annot=True, fmt=".2f", cmap="YlOrRd", vmin=1, vmax=10,
            cbar_kws={'label': 'Pairwise VIF (值域被限制在 1 ~ 10)'})
plt.title('Top 15 特征成对 VIF (共线性) 热力图\n(值越接近 1 说明特征间越独立)', fontsize=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('pictures/06_Top15_Pairwise_VIF_Heatmap.png', dpi=300)
plt.close()

# ================= 保存与输出结果 =================
# 1. 保存全量特征打分评估表
results_df.to_csv('feature_evaluation_scores.csv', encoding='utf-8-sig')

# 2. 保存只包含 Top15 特征和血糖的数据集用于建模
final_cols = top15_features + ['血糖']
df_final = df[final_cols]
df_final.to_csv('with_blood_top15.csv', index=False, encoding='utf-8-sig')

print("\n================= 集成评估完成 =================\n")
print("🎉 最终优胜的 15 个主要特征及其综合得分如下：")
print("-" * 50)
for i, feature in enumerate(top15_features, 1):
    score = results_df.loc[feature, '综合总分']
    print(f"{i:02d}. {feature:<20} \t综合得分: {score:.2f} / 400")
print("-" * 50)
print("\n所有图表(6张)已保存至 'pictures' 文件夹。")
print("全量特征打分明细已保存至 'feature_evaluation_scores.csv'。")
print("入模数据集已保存至 'with_blood_top15.csv'。")