import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import warnings

# 忽略警告并设置中文字体
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows用黑体
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac用户请取消注释这行
plt.rcParams['axes.unicode_minus'] = False


def comprehensive_feature_selection(data_path='cleaned_with_blood.csv'):
    print("🚀 开始执行多维度特征筛选综合评估...")

    # 1. 读取清洗后的训练集数据
    df = pd.read_csv(data_path)

    # 拆分特征(X)和目标变量(y)
    target_col = '血糖'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    features = X.columns.tolist()

    # 初始化一个 DataFrame 来保存三种方法的得分
    score_df = pd.DataFrame({'特征名称': features})

    # ==========================================
    # 方法一：Spearman 相关性分析 (非线性+防极端值)
    # ==========================================
    print("\n🔍 正在进行 方法一：Spearman 相关系数计算...")
    corr_matrix = df.corr(method='spearman')
    corr_with_target = corr_matrix[target_col].drop(target_col)

    score_df['相关性绝对值'] = corr_with_target.abs().values
    top_corr = corr_with_target.abs().sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(x=top_corr.values, y=top_corr.index, palette='viridis')
    # 给所有柱子加上数值标签（单位：相关系数）
    ax1.bar_label(ax1.containers[0], fmt='%.3f', padding=5)
    # 延长X轴防止文字被切掉
    ax1.set_xlim(0, max(top_corr.values) * 1.15)
    plt.title('Top 15 与血糖相关性最强的特征 (Spearman)', fontsize=16)
    plt.xlabel('Spearman 相关系数 (绝对值)')
    plt.tight_layout()
    plt.savefig('1_correlation_top15.png', dpi=300)
    plt.close()

    # ==========================================
    # 方法二：随机森林特征重要性 (非线性树模型)
    # ==========================================
    print("🌲 正在进行 方法二：随机森林特征重要性评估 (可能需要几秒钟)...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    score_df['随机森林重要性'] = rf.feature_importances_
    top_rf = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(x=top_rf.values, y=top_rf.index, palette='magma')
    # 给所有柱子加上数值标签（单位：Gini不纯度比率）
    ax2.bar_label(ax2.containers[0], fmt='%.4f Gini', padding=5)
    ax2.set_xlim(0, max(top_rf.values) * 1.15)
    plt.title('Top 15 随机森林特征重要性', fontsize=16)
    plt.xlabel('重要性得分 (Gini Importance)')
    plt.tight_layout()
    plt.savefig('2_random_forest_top15.png', dpi=300)
    plt.close()

    # ==========================================
    # 方法三：Lasso 回归正则化 (线性剔除)
    # ==========================================
    print("✂️ 正在进行 方法三：Lasso 回归特征惩罚与筛选...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_scaled, y)

    score_df['Lasso系数绝对值'] = np.abs(lasso.coef_)
    top_lasso = pd.Series(np.abs(lasso.coef_), index=features).sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    ax3 = sns.barplot(x=top_lasso.values, y=top_lasso.index, palette='crest')
    # 给所有柱子加上数值标签（单位：惩罚后系数）
    ax3.bar_label(ax3.containers[0], fmt='%.3f 系数', padding=5)
    ax3.set_xlim(0, max(top_lasso.values) * 1.15)
    plt.title('Top 15 Lasso回归核心特征 (剔除冗余)', fontsize=16)
    plt.xlabel('Lasso 系数绝对值 (惩罚后标准化)')
    plt.tight_layout()
    plt.savefig('3_lasso_top15.png', dpi=300)
    plt.close()

    # ==========================================
    # 终极奥义：多模型归一化综合打分
    # ==========================================
    print("🏆 正在计算 终极综合特征排行榜...")
    min_max_scaler = MinMaxScaler(feature_range=(0, 100))

    score_df['相关性_Score'] = min_max_scaler.fit_transform(score_df[['相关性绝对值']])
    score_df['随机森林_Score'] = min_max_scaler.fit_transform(score_df[['随机森林重要性']])
    score_df['Lasso_Score'] = min_max_scaler.fit_transform(score_df[['Lasso系数绝对值']])

    # 计算总分 (等权重 1:1:1，满分300)
    score_df['综合总分'] = score_df['相关性_Score'] + score_df['随机森林_Score'] + score_df['Lasso_Score']
    final_report = score_df.sort_values(by='综合总分', ascending=False).reset_index(drop=True)

    # 保存结果到 CSV
    final_report.to_csv('4_final_feature_ranking.csv', index=False, encoding='utf-8-sig')

    # ------------------------------------------
    # 新增：绘制第四张图 —— 综合总分排行榜
    # ------------------------------------------
    top_final = final_report.head(15)

    plt.figure(figsize=(10, 6))
    ax4 = sns.barplot(x=top_final['综合总分'], y=top_final['特征名称'], palette='rocket')
    # 给所有柱子加上数值标签（单位：分）
    ax4.bar_label(ax4.containers[0], fmt='%.2f 分', padding=5)
    # 因为满分是300分，我们稍微留一点余地到340，这样“分”字不会出界
    ax4.set_xlim(0, 340)
    plt.title('Top 15 终极核心特征排行榜 (多模型综合)', fontsize=16, fontweight='bold')
    plt.xlabel('综合总分 (满分300分)')
    plt.tight_layout()
    plt.savefig('4_comprehensive_top15.png', dpi=300)
    plt.close()

    print("\n🎉 === 最终筛选出的 Top 15 核心主要变量 ===")
    print(top_final[['特征名称', '综合总分']].to_string(index=False))

    print("\n✅ 所有计算完成！已生成以下文件：")
    print(" - 1_correlation_top15.png (相关性排行图)")
    print(" - 2_random_forest_top15.png (随机森林排行图)")
    print(" - 3_lasso_top15.png (Lasso排行图)")
    print(" - 4_comprehensive_top15.png (综合总分排行图 —— 【新增】)")
    print(" - 4_final_feature_ranking.csv (包含所有指标打分的终极排行榜表格)")


if __name__ == "__main__":
    comprehensive_feature_selection('cleaned_with_blood.csv')