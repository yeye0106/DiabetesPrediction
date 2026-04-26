import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings

# 忽略警告并设置中文字体
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows用黑体
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac用户请取消注释这行
plt.rcParams['axes.unicode_minus'] = False


def train_and_evaluate_models():
    print("🚀 开始进行 问题2：血糖值预测模型的构建与训练...")

    # 1. 读取清洗后的数据和上一问选出的 Top 15 特征
    data_df = pd.read_csv('cleaned_with_blood.csv')
    feature_df = pd.read_csv('4_final_feature_ranking.csv')

    # 提取前 15 名特征
    top_15_features = feature_df['特征名称'].head(15).tolist()
    print(f"📦 成功加载数据，使用 问题1 筛选出的 Top 15 核心特征进行建模：\n{top_15_features}")

    # 2. 划分特征(X)和标签(y)
    X = data_df[top_15_features]
    y = data_df['血糖']

    # 划分训练集 (80%) 和 测试集 (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✂️ 数据集划分完毕 -> 训练集: {X_train.shape[0]} 个样本, 测试集: {X_test.shape[0]} 个样本")

    # 3. 初始化三大主流回归模型
    models = {
        '随机森林 (Random Forest)': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1,
                                      verbose=-1)
    }

    results = []
    best_model_name = ""
    best_r2 = -float('inf')
    best_model = None
    best_y_pred = None

    print("\n⚔️ 开始模型擂台赛...")
    # 4. 训练与评估
    for name, model in models.items():
        # 训练模型
        model.fit(X_train, y_train)

        # 在测试集上进行预测
        y_pred = model.predict(X_test)

        # 计算评估指标
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        results.append({
            '模型名称': name,
            'R² (决定系数)': round(r2, 4),
            'RMSE (均方根误差)': round(rmse, 4),
            'MAE (平均绝对误差)': round(mae, 4)
        })

        print(f"✅ {name} 评估完成 -> R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # 记录表现最好(R2最高)的模型
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model
            best_y_pred = y_pred

    # 输出结果表格
    results_df = pd.DataFrame(results)
    results_df.to_csv('5_model_comparison_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n🏆 最终优胜模型是: 【{best_model_name}】")

    # 5. 数据可视化
    # --- 图 1：模型指标对比图 ---
    plt.figure(figsize=(12, 5))

    # R2对比图
    plt.subplot(1, 2, 1)
    ax1 = sns.barplot(x='R² (决定系数)', y='模型名称', data=results_df, palette='Blues_r')
    plt.title('各模型 R² 决定系数对比 (越高越好)', fontsize=14)
    plt.xlabel('R² Score')
    plt.ylabel('')
    for p in ax1.patches:
        ax1.text(p.get_width() + 0.01, p.get_y() + p.get_height() / 2., f'{p.get_width():.4f}', va='center')

    # RMSE对比图
    plt.subplot(1, 2, 2)
    ax2 = sns.barplot(x='RMSE (均方根误差)', y='模型名称', data=results_df, palette='Reds_r')
    plt.title('各模型 RMSE 均方根误差对比 (越低越好)', fontsize=14)
    plt.xlabel('RMSE')
    plt.ylabel('')
    for p in ax2.patches:
        ax2.text(p.get_width() + 0.01, p.get_y() + p.get_height() / 2., f'{p.get_width():.4f}', va='center')

    plt.tight_layout()
    plt.savefig('6_model_performance_comparison.png', dpi=300)
    plt.close()

    # --- 图 2：最佳模型的 真实值 vs 预测值 散点拟合图 ---
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_y_pred, alpha=0.5, color='teal', edgecolor='k')

    # 画一条完美的 y=x 对角线作为基准
    max_val = max(max(y_test), max(best_y_pred))
    min_val = min(min(y_test), min(best_y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线 (y=x)')

    plt.title(f'最佳模型 ({best_model_name}) : 真实血糖 vs 预测血糖', fontsize=15)
    plt.xlabel('真实血糖值 (Test Set)', fontsize=12)
    plt.ylabel('预测血糖值 (Predicted)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('7_best_model_scatter.png', dpi=300)
    plt.close()

    # --- 图 3：最佳模型的 残差分布图 ---
    residuals = y_test - best_y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='purple', bins=40)
    plt.title(f'最佳模型 ({best_model_name}) : 预测残差分布', fontsize=15)
    plt.xlabel('残差 (真实值 - 预测值)', fontsize=12)
    plt.ylabel('频数', fontsize=12)

    # 标注均值线
    plt.axvline(x=0, color='r', linestyle='--', label='零残差线')
    plt.legend()
    plt.tight_layout()
    plt.savefig('8_best_model_residuals.png', dpi=300)
    plt.close()

    print("\n✅ 建模完毕！生成文件如下：")
    print(" - 5_model_comparison_results.csv (各模型指标报表)")
    print(" - 6_model_performance_comparison.png (模型评价对比条形图)")
    print(" - 7_best_model_scatter.png (预测效果拟合图 - 极其重要)")
    print(" - 8_best_model_residuals.png (残差呈正态分布即证明模型合理)")

    # (额外福利) 如果你想保存最佳模型去跑最后一问，这里可以直接导出：
    import joblib
    joblib.dump(best_model, 'best_blood_sugar_model.pkl')
    print(" - best_blood_sugar_model.pkl (保存好的冠军模型，将在 问题4 中直接加载使用！)")


if __name__ == "__main__":
    train_and_evaluate_models()