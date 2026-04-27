import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import os
import warnings

warnings.filterwarnings('ignore')

# ================= 基础设置 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('pictures', exist_ok=True)

print("📥 加载预处理数据...")
df = pd.read_csv('with_blood_processed.csv')
y = df['血糖']
X = df.drop(columns=['血糖'])

# 结果收集容器
results_df = pd.DataFrame(index=X.columns)

# ================= 1. Pearson 初筛与打分 =================
print("🔍 步骤1: Pearson 相关系数初筛 (剔除 |r|<0.1)...")
pearson = df.corr()['血糖'].abs().drop('血糖')
results_df['Pearson'] = pearson

# 初筛过滤（题目要求）
pearson_pass = pearson[pearson >= 0.1].index.tolist()
print(f"✅ Pearson 初筛保留: {len(pearson_pass)} 个特征")

# 仅对通过初筛的特征进行后续计算
X_pass = X[pearson_pass]
results_df = results_df.loc[pearson_pass]

# ================= 2. VIF 共线性检验与打分 =================
print("📐 步骤2: VIF 多重共线性检验 (剔除 >10)...")
# 使用稳定的 OLS 拟合计算 VIF，避免矩阵求逆奇异
from numpy.linalg import pinv
corr_mat = X_pass.corr().values
vif_values = np.diag(pinv(corr_mat))
# 修正理论最小值 1.0
vif_values = np.maximum(vif_values, 1.0)
vif_series = pd.Series(vif_values, index=X_pass.columns)
results_df['VIF'] = vif_series

# 题目要求：>10 直接剔除
vif_pass = vif_series[vif_series <= 10].index.tolist()
print(f"✅ VIF 检验保留: {len(vif_pass)} 个特征 (剔除强共线性)")
results_df = results_df.loc[vif_pass]

# ================= 3. Lasso 正则化筛选 =================
print("📉 步骤3: Lasso 回归 (L1正则压缩)...")
# 注意：X_pass 已在预处理中标准化，此处无需重复 StandardScaler
lasso = LassoCV(cv=5, random_state=42, n_jobs=-1).fit(X_pass, y)
results_df['Lasso'] = pd.Series(np.abs(lasso.coef_), index=X_pass.columns)

# ================= 4. 随机森林重要性 =================
print("🌲 步骤4: Random Forest 特征重要性...")
rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_pass, y)
results_df['RF'] = pd.Series(rf.feature_importances_, index=X_pass.columns)

# ================= 5. 综合归一化与加权打分 =================
print("📊 步骤5: 综合归一化打分...")
minmax = MinMaxScaler(feature_range=(0, 100))

# 安全归一化（处理常数列防除零）
results_df['S_Pearson'] = minmax.fit_transform(results_df[['Pearson']].fillna(0))
results_df['S_VIF'] = minmax.fit_transform((1 / (results_df[['VIF']] + 1e-5)).fillna(0))
results_df['S_Lasso'] = minmax.fit_transform(results_df[['Lasso']].fillna(0))
results_df['S_RF'] = minmax.fit_transform(results_df[['RF']].fillna(0))

# 权重配置：模型驱动为主，统计为辅
results_df['综合得分'] = (0.2 * results_df['S_Pearson'] +
                          0.2 * results_df['S_VIF'] +
                          0.3 * results_df['S_Lasso'] +
                          0.3 * results_df['S_RF'])

# ================= 6. Top 15 强制保留机制 =================
print("🎯 步骤6: 选取 Top 15 特征...")
results_df = results_df.sort_values('综合得分', ascending=False)
top15 = results_df.head(15).copy()

# 强制保留医学划分指标（年龄、性别）
mandatory = ['年龄', '性别']
for m in mandatory:
    if m not in top15.index and m in results_df.index:
        top15.loc[m] = results_df.loc[m]
        # 若超出15个，剔除得分最低者
        if len(top15) > 15:
            top15 = top15.iloc[:-1]

final_features = top15.index.tolist()

# ================= 7. 可视化与输出 =================
print("📈 生成可视化图表...")
plt.figure(figsize=(10, 8))
sns.barplot(x=top15['综合得分'].values, y=top15.index, color='darkorange', edgecolor='black')
for i, v in enumerate(top15['综合得分'].values):
    plt.text(v + 2, i, f'{v:.1f}', va='center', fontsize=10)
plt.title('Top 15 特征综合评估得分', fontsize=15)
plt.xlabel('综合得分')
plt.tight_layout()
plt.savefig('pictures/07_Top15_Final_Score.png', dpi=300)
plt.close()

# 保存结果
results_df.to_csv('feature_evaluation_scores.csv', encoding='utf-8-sig')
df_final = df[final_features + ['血糖']]
df_final.to_csv('with_blood_top15.csv', index=False, encoding='utf-8-sig')

print("\n" + "="*50)
print("🏆 最终入模的 15 个主要特征如下：")
for i, feat in enumerate(final_features, 1):
    print(f"{i:02d}. {feat:<15} \t综合得分: {top15.loc[feat, '综合得分']:.2f}")
print("="*50)
print("✅ 已保存: with_blood_top15.csv (用于问题2/3/4训练)")