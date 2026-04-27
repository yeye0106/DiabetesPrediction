import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 读取数据 ====================
train_df = pd.read_csv('with_blood_processed.csv')
print("训练集形状:", train_df.shape)

# 分离 X 和 y
y = train_df['血糖']
X = train_df.drop('血糖', axis=1)
feature_names = X.columns.tolist()
print(f"候选特征数量: {len(feature_names)}")

# ==================== 2. Pearson 相关系数 ====================
pearson_corr = X.apply(lambda col: col.corr(y))
pearson_corr_abs = pearson_corr.abs().sort_values(ascending=False)

results = {
    'feature': feature_names,
    'pearson_corr': pearson_corr[feature_names].values,
    'pearson_abs': pearson_corr_abs[feature_names].values
}

# 柱状图
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
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_values = []
    for i in range(X.shape[1]):
        try:
            vif = variance_inflation_factor(X.values, i)
            if np.isinf(vif) or np.isnan(vif):
                vif = 1000
        except:
            vif = 1000
        vif_values.append(vif)
    vif_data['VIF'] = vif_values
    return vif_data

vif_result = calculate_vif(X)
vif_result = vif_result.sort_values('VIF', ascending=False)
results['vif'] = vif_result.set_index('feature')['VIF'][feature_names].values

# VIF 柱状图（原始）
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

# 对数尺度图
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

# ==================== 4. Lasso 回归 ====================
lasso = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-4, 1, 50))
lasso.fit(X, y)
lasso_coef = lasso.coef_
results['lasso_coef'] = lasso_coef

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

# ==================== 5. 随机森林 ====================
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
df_results = pd.DataFrame(results)
df_results.rename(columns={'vif': 'VIF'}, inplace=True)

df_results['pearson_rank'] = df_results['pearson_abs'].rank(ascending=False, method='min')
df_results['vif_rank'] = df_results['VIF'].rank(ascending=True, method='min')
df_results['lasso_rank'] = df_results['lasso_coef'].abs().rank(ascending=False, method='min')
df_results['rf_rank'] = df_results['rf_imp'].rank(ascending=False, method='min')

df_results['total_score'] = (df_results['pearson_rank'] +
                             df_results['vif_rank'] +
                             df_results['lasso_rank'] +
                             df_results['rf_rank'])
df_results = df_results.sort_values('total_score')

# 选出前15个特征
top15_features = df_results.head(15)['feature'].tolist()
print("\n===== 优胜的15个字段（技术排名）=====")
for i, f in enumerate(top15_features, 1):
    print(f"{i:2d}. {f}")

# 保存详细排名 CSV
df_results.to_csv('feature_selection_ranking.csv', index=False, encoding='utf-8-sig')
print("\n特征筛选排名表已保存为 'feature_selection_ranking.csv'")

# 输出前15个特征的值 CSV（含血糖）
top15_values = train_df[top15_features + ['血糖']]
top15_values.to_csv('top15_features_values.csv', index=False, encoding='utf-8-sig')
print("前15个特征的值（含血糖）已保存为 'top15_features_values.csv'")

# ==================== 7. 最终综合得分柱状图 ====================
top15_df = df_results.head(15).copy()
plt.figure(figsize=(12, 6))
plt.barh(top15_df['feature'], top15_df['total_score'], color='darkblue')
for i, v in enumerate(top15_df['total_score']):
    plt.text(v + 1, i, f"{v:.1f}", va='center')
plt.xlabel('综合得分（排名和，越低越好）')
plt.title('前15个特征的综合得分')
plt.tight_layout()
plt.savefig('pictures/final_top15_score.png', dpi=300, bbox_inches='tight')
plt.close()
print("最终综合得分柱状图已保存")

# ==================== 8. 特征间相关性热力图（15×15） ====================
X_top15 = X[top15_features]
corr_matrix = X_top15.corr()

plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('前15个特征之间的相关系数热力图', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('pictures/top15_corr_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("前15个特征相关性热力图已保存为 'top15_corr_heatmap.png'")

# 可选：输出这些特征的 VIF 值（仅控制台参考）
vif_top15 = calculate_vif(X_top15)
vif_top15 = vif_top15.set_index('feature')
print("\n前15个特征的 VIF 值：")
print(vif_top15.sort_values('VIF', ascending=False))