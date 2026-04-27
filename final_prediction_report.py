import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 0. 环境与配置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data_dir = 'processed_data'
model_dir = 'models'
res_dir = 'results'
pic_dir = 'pictures'
os.makedirs(res_dir, exist_ok=True)
os.makedirs(pic_dir, exist_ok=True)

# ==========================================
# 1. 加载数据与模型
# ==========================================
print("1. 正在加载预测数据与已训练模型...")
# 加载在预处理阶段已经完成“严格标准化”的预测集数据
test_X_preprocessed = pd.read_csv(os.path.join(data_dir, 'test_preprocessed.csv'))
# 加载原始预测集，用于提取原始年龄、性别和ID进行报告展示
raw_test = pd.read_csv('within_blood.csv')

# 加载保存的成果
reg_model = joblib.load(os.path.join(model_dir, 'best_blood_sugar_stacking_model.pkl'))
clf_model = joblib.load(os.path.join(model_dir, 'best_risk_classification_stacking.pkl'))
top_features = joblib.load(os.path.join(model_dir, 'model_features.pkl'))
best_threshold = joblib.load(os.path.join(model_dir, 'optimal_high_risk_threshold.pkl'))

# 确保特征对齐
X_target = test_X_preprocessed[top_features]

# ==========================================
# 2. 执行双重评估预测
# ==========================================
print("2. 正在执行血糖数值回归与风险概率评估...")

# A. 血糖值回归预测 (注意还原 Log1p)
pred_sugar_log = reg_model.predict(X_target)
pred_sugar_raw = np.expm1(pred_sugar_log)

# B. 糖尿病风险概率预测
pred_proba = clf_model.predict_proba(X_target)
P1, P2, P3 = pred_proba[:, 0], pred_proba[:, 1], pred_proba[:, 2]

# C. 应用黄金预警阈值 (P3 >= best_threshold 则判定为高风险)
final_risk_label = []
for i in range(len(P3)):
    if P3[i] >= best_threshold:
        final_risk_label.append('高风险')
    elif (P2[i] + P3[i]) >= 0.45:
        final_risk_label.append('中风险')
    else:
        final_risk_label.append('低风险')

# ==========================================
# 3. 整合结果报表
# ==========================================
print("3. 正在生成 141 名受检者详细预测报表...")
report_df = pd.DataFrame({
    'id': raw_test['id'],
    '性别': raw_test['性别'],
    '年龄': raw_test['年龄'],
    '预测血糖值(mmol/L)': np.round(pred_sugar_raw, 2),
    '低风险概率(P1)': np.round(P1, 4),
    '中风险概率(P2)': np.round(P2, 4),
    '高风险概率(P3)': np.round(P3, 4),
    '高风险评分(Score)': np.round(P3 * 100, 2),
    '最终风险评级': final_risk_label
})

# 保存详细 CSV 结果
csv_path = os.path.join(res_dir, 'final_test_prediction_results.csv')
report_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

# ==========================================
# 4. 人群画像统计分析 (划分年龄段与性别)
# ==========================================
print("4. 正在进行人群画像多维度统计...")

# 4.1 定义年龄段划分
age_bins = [0, 30, 45, 60, 100]
age_labels = ['青年(<30)', '壮年(30-45)', '中年(45-60)', '老年(>60)']
report_df['年龄段'] = pd.cut(report_df['年龄'], bins=age_bins, labels=age_labels)

# 4.2 统计分析
# 性别风险占比
summary_gender = report_df.groupby('性别', observed=True)['最终风险评级'].value_counts(normalize=True).unstack().fillna(0)
# 年龄段风险占比
summary_age = report_df.groupby('年龄段', observed=True)['最终风险评级'].value_counts(normalize=True).unstack().fillna(0)

# 打印控制台摘要
print("\n--- [性别维度] 风险分布画像 ---")
print((summary_gender * 100).round(2).astype(str) + '%')
print("\n--- [年龄维度] 风险分布画像 ---")
print((summary_age * 100).round(2).astype(str) + '%')

# ==========================================
# 5. 可视化重要数据图
# ==========================================
print("5. 正在导出人群画像统计图...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 图1：不同年龄段的风险结构堆叠图
summary_age.plot(kind='bar', stacked=True, ax=axes[0], color=['#2ecc71', '#f1c40f', '#e74c3c'], alpha=0.8)
axes[0].set_title('不同年龄段糖尿病风险画像分布', fontsize=14)
axes[0].set_ylabel('占比 (1.0=100%)')
axes[0].set_xlabel('年龄组')
axes[0].legend(title='风险等级', loc='upper right')
axes[0].tick_params(axis='x', rotation=0)

# 图2：不同性别的风险分布对比图
summary_gender.plot(kind='bar', stacked=True, ax=axes[1], color=['#2ecc71', '#f1c40f', '#e74c3c'], alpha=0.8)
axes[1].set_title('不同性别糖尿病风险占比对比', fontsize=14)
axes[1].set_ylabel('占比 (1.0=100%)')
axes[1].set_xlabel('性别')
axes[1].legend(title='风险等级', loc='upper right')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
pic_path = os.path.join(pic_dir, '10_Final_Population_Analysis.png')
plt.savefig(pic_path, dpi=300)

# ==========================================
# 6. 额外产出：预测血糖分布情况
# ==========================================
plt.figure(figsize=(10, 6))
sns.histplot(data=report_df, x='预测血糖值(mmol/L)', hue='最终风险评级', kde=True, palette='Set1', element='step')
plt.title('预测血糖数值分布及风险归类概览', fontsize=14)
plt.savefig(os.path.join(pic_dir, '11_Predicted_Glucose_Distribution.png'), dpi=300)

print("\n" + "="*50)
print("✅ 第四问所有任务圆满完成！")
print(f"1. 详细报表已保存至: {csv_path}")
print(f"2. 人群画像统计图已保存至: {pic_path}")
print(f"3. 血糖分布概览图已保存至: {os.path.join(pic_dir, '11_Predicted_Glucose_Distribution.png')}")
print("="*50)