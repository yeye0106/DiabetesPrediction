import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss, confusion_matrix, recall_score, precision_score

warnings.filterwarnings('ignore')

# ==========================================
# 0. 环境与配置初始化
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data_dir = 'processed_data'
model_dir = 'models'
pic_dir = 'pictures'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(pic_dir, exist_ok=True)

# ==========================================
# 1. 加载数据并生成多分类标签
# ==========================================
print("1. 正在加载数据...")
train_df = pd.read_csv(os.path.join(data_dir, 'train_preprocessed.csv'))

try:
    top_features = joblib.load(os.path.join(model_dir, 'model_features.pkl'))
except FileNotFoundError:
    top_features = [col for col in train_df.columns if col != '血糖']

X = train_df[top_features]

# 根据血糖值划分三类风险等级 (0:低, 1:中, 2:高)
y = pd.cut(train_df['血糖'], bins=[-np.inf, 6.1, 6.7, np.inf], labels=[0, 1, 2]).astype(int)

# 划分训练集和验证集 (80/20，必须使用分层抽样 stratify 保证比例一致)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 2. 核心算法重构：Cost-Sensitive Stacking
# ==========================================
print("\n2. 正在构建双重加权 Stacking 分类算法...")

# (1) 基础模型池：全部开启对抗极度不平衡的 class_weight
lgb = LGBMClassifier(
    n_estimators=300, learning_rate=0.01, max_depth=6, num_leaves=31,
    class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
)

rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=2,
    class_weight='balanced', random_state=42, n_jobs=-1
)

cb = CatBoostClassifier(
    iterations=400, learning_rate=0.02, depth=6,
    auto_class_weights='Balanced', random_state=42, verbose=0, thread_count=-1
)

# (2) 元模型设计：带惩罚项和类平衡的逻辑回归
# 它将学习三个基模型的输出概率，并自动纠正它们的系统性偏差
meta_learner = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

# (3) 组装 Stacking 模型 (使用 'predict_proba' 作为特征传递给元模型)
stacking_clf = StackingClassifier(
    estimators=[('lgb', lgb), ('rf', rf), ('cb', cb)],
    final_estimator=meta_learner,
    cv=5,  # 内部 5 折交叉验证防止过拟合
    n_jobs=-1,
    stack_method='predict_proba'
)

print("   -> 正在训练 Stacking 模型 (耗时可能稍长)...")
stacking_clf.fit(X_train, y_train)

# ==========================================
# 3. 提取概率矩阵
# ==========================================
print("\n3. 正在提取风险预测概率...")
y_pred_proba = stacking_clf.predict_proba(X_val)

P1 = y_pred_proba[:, 0]
P2 = y_pred_proba[:, 1]
P3 = y_pred_proba[:, 2]

# 评价 Log Loss
loss = log_loss(y_val, y_pred_proba)
print(f"✅ Stacking 交叉熵损失 (Log Loss): {loss:.4f}")

# ==========================================
# 4. 极致优化：动态阈值搜索 (找出最佳预警线)
# ==========================================
print("\n4. 正在执行动态阈值搜索算法 (最大化高风险 Recall)...")

# 我们的目标是：高危标签为 2
y_val_binary_high = (y_val == 2).astype(int)

best_threshold = 0.5
best_recall = 0
target_precision_min = 0.15  # 我们容忍一定程度的误报，但 Precision 不能崩盘到 0

# 遍历 0.10 到 0.60 的概率阈值
for thresh in np.arange(0.10, 0.61, 0.01):
    # 只要高风险概率大于阈值，就强制判定为 2 (高危)
    temp_pred = (P3 >= thresh).astype(int)

    rec = recall_score(y_val_binary_high, temp_pred)
    prec = precision_score(y_val_binary_high, temp_pred, zero_division=0)

    # 寻找在满足最低精确率要求下，召回率最高的黄金阈值
    if prec >= target_precision_min and rec > best_recall:
        best_recall = rec
        best_threshold = thresh

print(f"⭐ 算法寻优完成！找到最佳高风险预警阈值: P3 >= {best_threshold:.2f}")
print(f"   在此阈值下，高风险人群 Recall 可达: {best_recall:.4f}")

# ==========================================
# 5. 应用黄金阈值并输出最终评级
# ==========================================
# 覆盖死板的 argmax 逻辑：
# 规则1: P3 >= 最佳阈值 -> 高风险(2)
# 规则2: 如果不满足规则1，但 (P2 + P3) >= 0.45 -> 中风险(1)
# 规则3: 其余 -> 低风险(0)

y_pred_custom = np.zeros_like(y_val)
for i in range(len(P3)):
    if P3[i] >= best_threshold:
        y_pred_custom[i] = 2
    elif (P2[i] + P3[i]) >= 0.45:
        y_pred_custom[i] = 1
    else:
        y_pred_custom[i] = 0

# ==========================================
# 6. 最终性能报告与可视化
# ==========================================
print("\n" + "=" * 60)
print("🏆 Stacking + 动态阈值 最终分类报告")
print("=" * 60)
report = classification_report(y_val, y_pred_custom, target_names=['低风险(0)', '中风险(1)', '高风险(2)'])
print(report)

# 混淆矩阵绘图
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_val, y_pred_custom)
sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd',
            xticklabels=['低风险(预测)', '中风险(预测)', '高风险(预测)'],
            yticklabels=['低风险(真实)', '中风险(真实)', '高风险(真实)'])
plt.title(f'优化后风险分类混淆矩阵\n(使用黄金预警线: P3 >= {best_threshold:.2f})', fontsize=14)
plt.ylabel('真实标签', fontsize=12)
plt.xlabel('自定义阈值预测标签', fontsize=12)
plt.tight_layout()

pic_save_path = os.path.join(pic_dir, '8_Optimized_Classification_Confusion_Matrix.png')
plt.savefig(pic_save_path, dpi=300)
print(f"\n✅ 混淆矩阵图已保存至: {pic_save_path}")

# ==========================================
# 7. 保存最终模型与阈值信息
# ==========================================
# 我们不仅要保存模型，还要把找到的最佳阈值存下来，给第四问的预测集使用
model_save_path = os.path.join(model_dir, 'best_risk_classification_stacking.pkl')
joblib.dump(stacking_clf, model_save_path)
joblib.dump(best_threshold, os.path.join(model_dir, 'optimal_high_risk_threshold.pkl'))
print(f"✅ 最优 Stacking 分类模型及动态阈值已保存！")