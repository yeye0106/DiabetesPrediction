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


def add_bar_labels(ax, format_str='{:.4f}'):
    """为横向柱状图添加数值标签的辅助函数"""
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(format_str.format(width),
                    (width, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center',
                    xytext=(5, 0),  # 向右偏移5个像素
                    textcoords='offset points',
                    fontsize=10)
    # 动态扩展X轴最大值，防止最长柱子的数值被图片边缘裁剪
    max_width = max([p.get_width() for p in ax.patches])
    ax.set_xlim(0, max_width * 1.15)


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

    # 绘制相关性 Top 15 柱状图
    top_corr = corr_with_target.abs().sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(x=top_corr.values, y=top_corr.index, palette='viridis')
    add_bar_labels(ax1, '{:.4f}')  # 添加4位小数标签
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

    # 绘制随机森林 Top 15 柱状图
    top_rf = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(x=top_rf.values, y=top_rf.index, palette='magma')
    add_bar_labels(ax2, '{:.5f}')  # 添加5位小数标签
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

    # 绘制 Lasso Top 15 柱状图
    top_lasso = pd.Series(np.abs(lasso.coef_), index=features).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    ax3 = sns.barplot(x=top_lasso.values, y=top_lasso.index, palette='crest')
    add_bar_labels(ax3, '{:.4f}')
    plt.title('Top 15 Lasso回归核心特征 (剔除冗余)', fontsize=16)
    plt.xlabel('Lasso 系数绝对值 (惩罚后)')
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

    score_df['综合总分'] = score_df['相关性_Score'] + score_df['随机森林_Score'] + score_df['Lasso_Score']
    final_report = score_df.sort_values(by='综合总分', ascending=False).reset_index(drop=True)
    final_report.to_csv('feature_ranking_report.csv', index=False, encoding='utf-8-sig')

    # 绘制第四张图：最终得分 Top 15 柱状图
    print("🎨 正在绘制 第四张终极得分图...")
    top_final = final_report.head(15)
    plt.figure(figsize=(10, 6))
    ax4 = sns.barplot(x='综合总分', y='特征名称', data=top_final, palette='rocket')
    add_bar_labels(ax4, '{:.2f}')  # 综合得分保留2位小数
    plt.title('Top 15 预测血糖核心变量 (多模型综合打分)', fontsize=16, fontweight='bold')
    plt.xlabel('综合总分 (满分300分)')
    plt.tight_layout()
    plt.savefig('4_final_score_top15.png', dpi=300)
    plt.close()

    print("\n🎉 === 最终筛选出的 Top 15 核心主要变量 ===")
    print(top_final[['特征名称', '综合总分']].to_string(index=False))

    print("\n✅ 所有计算完成！已生成以下文件：")
    print(" - 1_correlation_top15.png")
    print(" - 2_random_forest_top15.png")
    print(" - 3_lasso_top15.png")
    print(" - 4_final_score_top15.png (🌟 新增：综合得分可视化图表)")
    print(" - feature_ranking_report.csv")


if __name__ == "__main__":
    comprehensive_feature_selection('cleaned_with_blood.csv')