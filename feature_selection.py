import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
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

# ================= 第一步：Pearson 相关系数初筛 =================
print("\n--- 步骤 1: Pearson 相关系数初筛 ---")
# 计算所有特征与血糖的皮尔逊相关系数
corr_matrix = df.corr()
corr_with_target = corr_matrix['血糖'].drop('血糖').abs().sort_values(ascending=False)

# 绘制与血糖相关的Top20特征条形图
plt.figure(figsize=(10, 8))
sns.barplot(x=corr_with_target.head(20).values, y=corr_with_target.head(20).index, palette='viridis')
plt.title('Top 20 特征与血糖的 Pearson 相关系数 (绝对值)', fontsize=14)
plt.xlabel('Pearson Correlation (Absolute)')
plt.tight_layout()
plt.savefig('pictures/01_pearson_correlation.png', dpi=300)
plt.close()

# 为了防止遗漏潜在的非线性强特征，我们设定一个较宽容的初筛阈值（如选取 Top 25 供后续筛选）
selected_features_pearson = corr_with_target.head(25).index.tolist()
X_step1 = X[selected_features_pearson]
print(f"Pearson初筛保留特征数量: {len(selected_features_pearson)}")


# ================= 第二步：VIF 多重共线性检验 =================
print("\n--- 步骤 2: VIF 多重共线性检验 ---")
def calculate_vif(X_df):
    X_with_const = sm.add_constant(X_df)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_df.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i+1) for i in range(X_df.shape[1])]
    return vif_data

# 迭代剔除 VIF > 10 的强共线性特征
X_step2 = X_step1.copy()
while True:
    vif_df = calculate_vif(X_step2)
    max_vif = vif_df['VIF'].max()
    if max_vif > 10:
        # 找出 VIF 最高的特征并剔除
        feature_to_drop = vif_df.loc[vif_df['VIF'].idxmax(), 'Feature']
        print(f"剔除强共线性特征: {feature_to_drop} (VIF = {max_vif:.2f})")
        X_step2 = X_step2.drop(columns=[feature_to_drop])
    else:
        break

selected_features_vif = X_step2.columns.tolist()
print(f"VIF检验后保留特征数量: {len(selected_features_vif)}")


# ================= 第三步：Lasso 回归特征压缩 =================
print("\n--- 步骤 3: Lasso 回归 (L1正则化) ---")
# 使用交叉验证自动寻找最优的 alpha (惩罚系数)
lasso = LassoCV(cv=5, random_state=42).fit(X_step2, y)

# 提取非零系数的特征
lasso_coefs = pd.Series(lasso.coef_, index=X_step2.columns)
selected_features_lasso = lasso_coefs[lasso_coefs != 0].index.tolist()
print(f"Lasso 回归后保留特征数量: {len(selected_features_lasso)}")

# 绘制 Lasso 系数图
plt.figure(figsize=(10, 6))
lasso_coefs[lasso_coefs != 0].sort_values().plot(kind='barh', color='teal')
plt.title('Lasso 回归非零特征系数', fontsize=14)
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('pictures/02_lasso_coefficients.png', dpi=300)
plt.close()

X_step3 = X_step2[selected_features_lasso]


# ================= 第四步：随机森林特征重要性分析 =================
print("\n--- 步骤 4: 随机森林特征重要性提取 ---")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_step3, y)

rf_importances = pd.Series(rf.feature_importances_, index=X_step3.columns).sort_values(ascending=False)

# 截取最终的 Top 15 特征
final_15_features = rf_importances.head(15)

# 绘制最终 Top 15 特征重要性图
plt.figure(figsize=(10, 8))
sns.barplot(x=final_15_features.values, y=final_15_features.index, palette='magma')
plt.title('最终保留的 Top 15 核心建模特征 (基于 Random Forest)', fontsize=14)
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.savefig('pictures/03_final_15_features_rf.png', dpi=300)
plt.close()

print("\n================= 最终筛选结果 =================")
print("最终保留的 15 个主要变量如下（按重要性降序）：")
for i, (feat, imp) in enumerate(final_15_features.items(), 1):
    print(f"{i}. {feat} (重要度: {imp:.4f})")

# 将最终的 15 个特征连同血糖列保存为新的建模数据集
final_cols = final_15_features.index.tolist() + ['血糖']
df_final = df[final_cols]
df_final.to_csv('with_blood_top15.csv', index=False, encoding='utf-8-sig')
print("\n特征筛选完成！包含 15 个主要变量的新数据集已保存为 'with_blood_top15.csv'。")
print("相关可视化图表已保存至 'pictures' 文件夹。")