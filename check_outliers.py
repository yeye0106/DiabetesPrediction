import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 忽略警告并设置中文字体，防止图表中的中文显示为方块
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows用黑体
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac用户请使用这一行
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def detect_outliers_iqr(file_path, output_report_path, output_image_path, dataset_title):
    print(f"\n🚀 开始进行异常值检测 (基于 IQR 方法) - {dataset_title}...")

    # 1. 读取数据 (按要求直接读取，不加 encoding)
    df = pd.read_csv(file_path)

    # 2. 排除无需检测的列 (ID, 日期, 类别型, 以及缺失率极高的乙肝五项)
    cols_to_exclude = ['id', '体检日期', '性别', '乙肝e抗体', '乙肝e抗原', '乙肝核心抗体', '乙肝表面抗体',
                       '乙肝表面抗原']
    check_cols = [c for c in df.columns if c not in cols_to_exclude]

    report_data = []

    # 3. 遍历特征进行检测
    for col in check_cols:
        # 强制转换为数值型，非数值转为 NaN
        s = pd.to_numeric(df[col], errors='coerce')
        valid_s = s.dropna()

        if len(valid_s) == 0:
            continue

        # 计算四分位数和边界
        q1 = valid_s.quantile(0.25)
        q3 = valid_s.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # 找出异常值
        outliers = valid_s[(valid_s < lower_bound) | (valid_s > upper_bound)]
        outlier_count = len(outliers)
        valid_count = len(valid_s)
        outlier_pct = (outlier_count / valid_count) * 100 if valid_count > 0 else 0

        report_data.append({
            '特征名称': col,
            '有效样本数': valid_count,
            '异常值数量': outlier_count,
            '异常值占比(%)': round(outlier_pct, 2),
            '正常下界': round(lower_bound, 2),
            '正常上界': round(upper_bound, 2),
            '最小异常值': round(outliers.min(), 2) if outlier_count > 0 else None,
            '最大异常值': round(outliers.max(), 2) if outlier_count > 0 else None
        })

    # 4. 生成报告并按异常值数量降序排序
    report_df = pd.DataFrame(report_data)
    report_df = report_df.sort_values(by='异常值数量', ascending=False).reset_index(drop=True)

    # 打印异常值所有的特征
    print(f"\n📊 {dataset_title} 异常值所有的特征：")
    print(report_df.to_string(index=False))

    # 导出完整报告
    report_df.to_csv(output_report_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 完整异常值检测报告已保存至: '{output_report_path}'")

    # 5. 可视化：绘制异常值最多的前 8 个特征的箱线图
    top_8_features = report_df.head(8)['特征名称'].tolist()

    print(f"🎨 正在绘制 {dataset_title} 异常值分布箱线图...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 【新增】为整个图表设置主标题
    fig.suptitle(f"【{dataset_title}】 - 异常值最多的前8个特征分布", fontsize=22, fontweight='bold')

    axes = axes.flatten()

    for i, col in enumerate(top_8_features):
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        sns.boxplot(y=s, ax=axes[i], color='skyblue', width=0.4, fliersize=3)
        axes[i].set_title(f"{col}\n(异常率: {report_df.loc[i, '异常值占比(%)']}%)", fontsize=12)
        axes[i].set_ylabel("数值")

    plt.tight_layout()
    # 调整布局，防止主标题和子图标题重叠
    plt.subplots_adjust(top=0.90)

    plt.savefig(output_image_path, dpi=300)
    print(f"✅ 箱线图已保存至: '{output_image_path}'\n" + "=" * 50)
    plt.show()


# ================= 运行调用 =================
if __name__ == '__main__':
    # 1. 对训练集进行异常值检测
    detect_outliers_iqr(
        file_path='with_blood.csv',
        output_report_path='train_outliers_report.csv',
        output_image_path='train_outliers_boxplot.png',
        dataset_title='训练集 (with_blood)'
    )

    # 2. 对测试集进行异常值检测
    detect_outliers_iqr(
        file_path='within_blood.csv',
        output_report_path='test_outliers_report.csv',
        output_image_path='test_outliers_boxplot.png',
        dataset_title='测试集 (within_blood)'
    )