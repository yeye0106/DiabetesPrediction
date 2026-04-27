import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 读取数据 ====================
train_df = pd.read_csv('with_blood_processed.csv')
print("训练集形状:", train_df.shape)
print("特征列表:", train_df.columns.tolist())

# 分离 X 和 y
y = train_df['血糖']
X = train_df.drop('血糖', axis=1)
feature_names = X.columns.tolist()
print(f"候选特征数量: {len(feature_names)}")

# ==================== 2. Pearson 相关系数 ====================
pearson_corr = X.apply(lambda col: col.corr(y))
pearson_corr_abs = pearson_corr.abs().sort_values(ascending=False)

# 保存结果字典
results = {
    'feature': feature_names,
    'pearson_corr': pearson_corr[feature_names].values,
    'pearson_abs': pearson_corr_abs[feature_names].values
}

# 相关系数柱状图（绝对值）
plt.figure(figsize=(12, 6))
sorted_idx = np.argsort(results['pearson_abs'])
plt.barh(np.array(feature_names)[sorted_idx], results['pearson_abs'][sorted_idx], color='steelblue')
for i, v in enumerate(results['pearson_abs'][sorted_idx]):
    plt.text(v + 0.01, i, f"{v:.3f}", va='center')
plt.xlabel('Pearson 相关系数绝对值')
plt.title('Pearson 相关性（与血糖）')
plt.tight_layout()
plt.savefig('pictures/pearson_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Pearson 相关系数图已保存")

# ==================== 3. VIF 多重共线性检验 ====================
# 注意：VIF 计算需要标准化后的数据（我们已经标准化过）
# 对于每个特征，VIF = 1 / (1 - R^2)，R^2 是该特征与其他特征的回归决定系数
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    # 先计算一次，可能会因为共线严重导致奇异矩阵，采用逐步剔除高VIF特征的方式迭代
    # 为了最终得到所有特征的VIF（不考虑剔除），我们直接计算，如果出现奇异则增加伪计数
    vif_values = []
    for i in range(X.shape[1]):
        try:
            vif = variance_inflation_factor(X.values, i)
        except:
            # 如果计算失败（如奇异矩阵），使用高VIF标记
            vif = 1000
        vif_values.append(vif)
    vif_data['VIF'] = vif_values
    return vif_data

vif_result = calculate_vif(X)
vif_result = vif_result.sort_values('VIF', ascending=False)
results['vif'] = vif_result.set_index('feature')['VIF'][feature_names].values

# VIF 柱状图（取对数尺度和原始尺度两个图）
plt.figure(figsize=(12, 6))
sorted_vif = vif_result.sort_values('VIF')
plt.barh(sorted_vif['feature'], sorted_vif['VIF'], color='coral')
for i, v in enumerate(sorted_vif['VIF']):
    plt.text(v + 1, i, f"{v:.1f}", va='center')
plt.xlabel('VIF 值')
plt.title('多重共线性检验 (VIF)')
plt.tight_layout()
plt.savefig('pictures/vif_barplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 为了更好展示小VIF的差异，再做一个对数尺度图
plt.figure(figsize=(12, 6))
plt.barh(sorted_vif['feature'], np.log1p(sorted_vif['VIF']), color='coral')
for i, v in enumerate(sorted_vif['VIF']):
    plt.text(np.log1p(v) + 0.1, i, f"{v:.1f}", va='center')
plt.xlabel('log(1+VIF)')
plt.title('多重共线性检验 (VIF 对数尺度)')
plt.tight_layout()
plt.savefig('pictures/vif_log_barplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("VIF 图已保存")

# ==================== 4. Lasso 回归 (L1正则化) ====================
# LassoCV 自动选择 alpha
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X, y)
lasso_coef = lasso.coef_
results['lasso_coef'] = lasso_coef

# 柱状图（系数绝对值）
lasso_abs = np.abs(lasso_coef)
plt.figure(figsize=(12, 6))
sorted_lasso = np.argsort(lasso_abs)
plt.barh(np.array(feature_names)[sorted_lasso], lasso_abs[sorted_lasso], color='green')
for i, v in enumerate(lasso_abs[sorted_lasso]):
    plt.text(v + 0.01, i, f"{v:.4f}", va='center')
plt.xlabel('Lasso 系数绝对值')
plt.title('Lasso 回归特征选择 (L1正则化)')
plt.tight_layout()
plt.savefig('pictures/lasso_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Lasso 图已保存")

# ==================== 5. 随机森林特征重要性 ====================
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X, y)
rf_importance = rf.feature_importances_
results['rf_imp'] = rf_importance

plt.figure(figsize=(12, 6))
sorted_rf = np.argsort(rf_importance)
plt.barh(np.array(feature_names)[sorted_rf], rf_importance[sorted_rf], color='purple')
for i, v in enumerate(rf_importance[sorted_rf]):
    plt.text(v + 0.002, i, f"{v:.4f}", va='center')
plt.xlabel('随机森林特征重要性')
plt.title('随机森林特征重要性')
plt.tight_layout()
plt.savefig('pictures/rf_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("随机森林特征重要性图已保存")

# ==================== 6. 综合评分与排序 ====================
# 构建DataFrame
df_results = pd.DataFrame(results)
# 对于每个指标，转换为排名（越小越好的指标：VIF 越小越好；其他指标越大越好）
# 定义方向：pearson_abs, lasso_abs, rf_imp 都是越大越好；VIF 越小越好
df_results['pearson_rank'] = df_results['pearson_abs'].rank(ascending=False, method='min')
df_results['vif_rank'] = df_results['VIF'].rank(ascending=True, method='min')
df_results['lasso_rank'] = df_results['lasso_coef'].abs().rank(ascending=False, method='min')
df_results['rf_rank'] = df_results['rf_imp'].rank(ascending=False, method='min')

# 综合得分 = 四个排名的加权和（此处等权重，也可以根据需求调整）
weights = {'pearson': 1, 'vif': 1, 'lasso': 1, 'rf': 1}
df_results['total_score'] = (weights['pearson'] * df_results['pearson_rank'] +
                             weights['vif'] * df_results['vif_rank'] +
                             weights['lasso'] * df_results['lasso_rank'] +
                             weights['rf'] * df_results['rf_rank'])
df_results = df_results.sort_values('total_score')

# 选出前15个特征
top15_features = df_results.head(15)['feature'].tolist()
print("\n===== 优胜的15个字段 =====")
for i, f in enumerate(top15_features, 1):
    print(f"{i:2d}. {f}")

# 保存详细CSV
df_results.to_csv('feature_selection_ranking.csv', index=False, encoding='utf-8-sig')
print("\n特征筛选排名表已保存为 'feature_selection_ranking.csv'")

# ==================== 7. 综合得分柱状图（前15） ====================
top15_df = df_results.head(15).copy()
plt.figure(figsize=(12, 6))
plt.barh(top15_df['feature'], top15_df['total_score'], color='darkblue')
for i, v in enumerate(top15_df['total_score']):
    plt.text(v + 1, i, f"{v:.1f}", va='center')
plt.xlabel('综合得分 (排名和)')
plt.title('前15个特征的综合得分（得分越低越重要）')
plt.tight_layout()
plt.savefig('pictures/final_top15_score.png', dpi=300, bbox_inches='tight')
plt.close()
print("最终综合得分柱状图已保存")

# ==================== 8. VIF 热力图（仅针对最终15个特征） ====================
# 计算这15个特征的相关系数矩阵（因为VIF本身是多重共线性，热力图展示相关性更直观）
# 但我们这里展示相关性矩阵的热力图（值范围-1~1），但题目要求展示VIF热力图？实际上VIF是单变量指标。
# 为了更好地展示共线性结构，我们计算这15个特征的相关系数矩阵并绘制热力图。
X_top15 = X[top15_features]
corr_matrix = X_top15.corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('前15个特征的相关系数热力图（共线性可视化）')
plt.tight_layout()
plt.savefig('pictures/top15_corr_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("前15个特征相关系数热力图已保存")

# 同时可以输出这些特征的VIF值（仅关注这15个）
vif_top15 = calculate_vif(X_top15)
print("\n前15个特征的VIF值：")
print(vif_top15.sort_values('VIF', ascending=False))

# ==================== 9. 额外输出优胜特征列表供后续使用 ====================
with open('selected_features.txt', 'w') as f:
    for feat in top15_features:
        f.write(feat + '\n')
print("所选特征列表已保存至 selected_features.txt")