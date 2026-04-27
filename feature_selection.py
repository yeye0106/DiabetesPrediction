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


# 辅助函数：绘制带数据标签的水平条形图（修正颜色与边界）
def plot_bar_with_labels(series, title, filename, unit='', top_n=20, color='steelblue', ascending=False):
    plt.figure(figsize=(10, 8))
    data_to_plot = series.sort_values(ascending=ascending).head(top_n)

    # 使用单一颜色，避免 palette 参数错误
    ax = sns.barplot(x=data_to_plot.values, y=data_to_plot.index, color=color, edgecolor='black')

    # 添加数据标签
    for container in ax.containers:
        ax.bar_label(container, fmt=f'%.4f {unit}', padding=5, fontsize=10)

    plt.title(title, fontsize=15, pad=15)
    plt.xlabel('数值 / 分数')
    plt.ylabel('特征名称')
    max_val = data_to_plot.max()
    # 处理全零情况
    if max_val > 0:
        plt.xlim(0 if data_to_plot.min() >= 0 else data_to_plot.min() * 1.1, max_val * 1.2)
    else:
        plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(f'pictures/{filename}', dpi=300)
    plt.close()


# ================= 第一步：Pearson 相关系数 =================
print("正在计算 Pearson 相关系数...")
corr_matrix = df.corr()
pearson_raw = corr_matrix['血糖'].drop('血糖').abs()
results_df['Pearson(绝对值)'] = pearson_raw

plot_bar_with_labels(pearson_raw, 'Top 20 Pearson相关系数 (绝对值)', '01_Pearson_Raw.png', color='cornflowerblue')

# ---------- 新增：Pearson初筛，剔除弱相关特征，但强制保留性别、年龄 ----------
print("正在进行 Pearson 初筛 (|r| >= 0.05，并强制保留性别、年龄)...")
# 初筛条件：相关系数绝对值 >= 0.05 或者特征名为 '性别' 或 '年龄'
selected_by_pearson = pearson_raw[(pearson_raw >= 0.05) | (pearson_raw.index.isin(['性别', '年龄']))].index.tolist()
print(f"初筛后保留特征数: {len(selected_by_pearson)}")
X = X[selected_by_pearson]   # 后续只对通过初筛的特征进行分析

# ================= 第二步：VIF 共线性计算 =================
print("正在计算 VIF 多重共线性...")
corr_mat = X.corr().values
inv_corr = np.linalg.pinv(corr_mat)
vif_raw = pd.Series(np.diag(inv_corr), index=X.columns)
vif_raw = vif_raw.apply(lambda x: max(1.0, x))   # 将小于1的值修正为1（理论最小值）
# 注意：results_df 中可能存在未通过初筛的特征，VIF 只对初筛后的特征填充
results_df['VIF值'] = np.nan
results_df.loc[vif_raw.index, 'VIF值'] = vif_raw

plot_bar_with_labels(vif_raw, 'Top 20 独立特征 (VIF值最小)', '02_VIF_Raw.png', ascending=True, color='mediumseagreen')


# ================= 第三步：Lasso 回归 =================
print("正在计算 Lasso 惩罚系数...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = LassoCV(cv=5, random_state=42, n_jobs=-1).fit(X_scaled, y)
lasso_raw = pd.Series(np.abs(lasso.coef_), index=X.columns)
results_df['Lasso(绝对系数)'] = np.nan
results_df.loc[lasso_raw.index, 'Lasso(绝对系数)'] = lasso_raw

plot_bar_with_labels(lasso_raw, 'Top 20 Lasso回归系数 (绝对值)', '03_Lasso_Raw.png', color='salmon')


# ================= 第四步：随机森林重要性 =================
print("正在计算 Random Forest 特征重要性...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)
rf_raw = pd.Series(rf.feature_importances_, index=X.columns)
results_df['RF(重要度)'] = np.nan
results_df.loc[rf_raw.index, 'RF(重要度)'] = rf_raw

plot_bar_with_labels(rf_raw, 'Top 20 随机森林特征重要性', '04_RF_Raw.png', color='orchid')


# ================= 第五步：综合打分与排名 =================
print("正在进行综合归一化打分...")
# 仅对通过初筛的特征进行打分，避免 NaN 干扰
scoring_features = X.columns.tolist()
minmax = MinMaxScaler(feature_range=(0, 100))

results_df['Pearson_Score'] = np.nan
results_df['VIF_Score'] = np.nan
results_df['Lasso_Score'] = np.nan
results_df['RF_Score'] = np.nan

results_df.loc[scoring_features, 'Pearson_Score'] = minmax.fit_transform(results_df.loc[scoring_features, ['Pearson(绝对值)']])
results_df.loc[scoring_features, 'VIF_Score'] = minmax.fit_transform(1 / results_df.loc[scoring_features, ['VIF值']])
results_df.loc[scoring_features, 'Lasso_Score'] = minmax.fit_transform(results_df.loc[scoring_features, ['Lasso(绝对系数)']])
results_df.loc[scoring_features, 'RF_Score'] = minmax.fit_transform(results_df.loc[scoring_features, ['RF(重要度)']])

# 等权重求和（满分400）
results_df['综合总分'] = (results_df['Pearson_Score'] + results_df['VIF_Score'] +
                         results_df['Lasso_Score'] + results_df['RF_Score'])

# ---------- 强制保留性别、年龄：将其综合总分设为满分，确保入选 ----------
print("强制保留性别、年龄特征...")
results_df.loc[['性别', '年龄'], '综合总分'] = 400.0

results_df = results_df.sort_values(by='综合总分', ascending=False)

plot_bar_with_labels(results_df['综合总分'].dropna(), '特征综合评估总分排名 (Top 20)', '05_Final_Total_Score.png',
                     unit='分', color='darkorange')


# ================= 第六步：Top 15 成对 VIF 热力图 =================
print("正在绘制 Top 15 成对 VIF 热力图...")
top15_features = results_df.head(15).index.tolist()
X_top15 = X[top15_features]   # 注意：这里特征可能超出 X 的范围吗？top15肯定在 X 中因为性别年龄已入选且其他特征也在初筛中

R_matrix = X_top15.corr().values
pairwise_vif = 1 / (1 - R_matrix ** 2 + 1e-5)   # 加小量防除零
pairwise_vif_df = pd.DataFrame(pairwise_vif, index=top15_features, columns=top15_features)

plt.figure(figsize=(12, 10))
sns.heatmap(pairwise_vif_df, annot=True, fmt=".2f", cmap="YlOrRd", vmin=1, vmax=10,
            cbar_kws={'label': 'Pairwise VIF (限制 1~10)'})
plt.title('Top 15 特征成对 VIF (共线性) 热力图\n(值越接近 1 说明特征间越独立)', fontsize=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('pictures/06_Top15_Pairwise_VIF_Heatmap.png', dpi=300)
plt.close()

# ================= 保存与输出结果 =================
results_df.to_csv('feature_evaluation_scores.csv', encoding='utf-8-sig')

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