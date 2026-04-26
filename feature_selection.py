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


def draw_chart_with_labels(x_data, y_data, title, xlabel, file_name, fmt_string, palette):
    """
    通用绘图函数：绘制水平柱状图并在每根柱子后方添加带有单位的数值标签
    """
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=x_data, y=y_data, palette=palette)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('特征名称', fontsize=12)

    # 遍历每根柱子，添加带有数值和单位的文本
    max_x = max(x_data)
    for p in ax.patches:
        width = p.get_width()
        # 格式化文本，例如 "300.00 分"
        label_text = fmt_string % width

        # 在柱子右侧偏移一点点的位置写上数值
        ax.text(width + (max_x * 0.015),
                p.get_y() + p.get_height() / 2.,
                label_text,
                ha="left", va="center",
                fontsize=11, fontweight='bold', color='black')

    # 动态扩展X轴，留出15%-20%的空白，防止文字被边缘切掉
    plt.xlim(0, max_x * 1.20)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.close()


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

    draw_chart_with_labels(
        x_data=top_corr.values,
        y_data=top_corr.index,
        title='图1：Top 15 与血糖相关性最强的特征 (Spearman)',
        xlabel='Spearman 相关系数 (绝对值)',
        file_name='picture/1_correlation_top15.png',
        fmt_string='%.3f (系数)',
        palette='viridis'
    )

    # ==========================================
    # 方法二：随机森林特征重要性 (非线性树模型)
    # ==========================================
    print("🌲 正在进行 方法二：随机森林特征重要性评估 (可能需要几秒钟)...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    score_df['随机森林重要性'] = rf.feature_importances_
    top_rf = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False).head(15)

    draw_chart_with_labels(
        x_data=top_rf.values,
        y_data=top_rf.index,
        title='图2：Top 15 随机森林特征重要性',
        xlabel='重要性得分 (Gini Importance)',
        file_name='picture/2_random_forest_top15.png',
        fmt_string='%.4f (重要度)',
        palette='magma'
    )

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

    draw_chart_with_labels(
        x_data=top_lasso.values,
        y_data=top_lasso.index,
        title='图3：Top 15 Lasso回归核心特征 (剔除冗余)',
        xlabel='Lasso 系数绝对值 (惩罚后)',
        file_name='picture/3_lasso_top15.png',
        fmt_string='%.3f (绝对值)',
        palette='crest'
    )

    # ==========================================
    # 终极奥义：多模型归一化综合打分
    # ==========================================
    print("🏆 正在计算 终极综合特征排行榜...")
    min_max_scaler = MinMaxScaler(feature_range=(0, 100))

    score_df['相关性_Score'] = min_max_scaler.fit_transform(score_df[['相关性绝对值']])
    score_df['随机森林_Score'] = min_max_scaler.fit_transform(score_df[['随机森林重要性']])
    score_df['Lasso_Score'] = min_max_scaler.fit_transform(score_df[['Lasso系数绝对值']])

    # 计算总分
    score_df['综合总分'] = score_df['相关性_Score'] + score_df['随机森林_Score'] + score_df['Lasso_Score']

    # 排序并提取前15名
    final_report = score_df.sort_values(by='综合总分', ascending=False).reset_index(drop=True)
    top_final = final_report.head(15)

    # 保存结果到 CSV
    final_report.to_csv('4_final_feature_ranking.csv', index=False, encoding='utf-8-sig')

    # 新增绘制：第 4 张 综合打分柱状图
    draw_chart_with_labels(
        x_data=top_final['综合总分'],
        y_data=top_final['特征名称'],
        title='图4：Top 15 终极核心变量综合打分',
        xlabel='多模型综合加权得分 (满分300分)',
        file_name='picture/4_final_score_top15.png',
        fmt_string='%.2f 分',
        palette='rocket'
    )

    print("\n🎉 === 最终筛选出的 Top 15 核心主要变量 ===")
    print(top_final[['特征名称', '综合总分']].to_string(index=False))

    print("\n✅ 所有计算完成！已生成以下文件：")
    print(" - 1_correlation_top15.png (相关性排行图)")
    print(" - 2_random_forest_top15.png (随机森林排行图)")
    print(" - 3_lasso_top15.png (Lasso排行图)")
    print(" - 4_final_score_top15.png  (🏆 新增：终极得分排行图)")
    print(" - 4_final_feature_ranking.csv (总榜单)")


if __name__ == "__main__":
    comprehensive_feature_selection('cleaned_with_blood.csv')