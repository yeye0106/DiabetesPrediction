import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import warnings

warnings.filterwarnings('ignore')


def preprocess_diabetes_data(train_path, test_path, out_train_path, out_test_path):
    print("🚀 开始执行超级高质量数据预处理...")

    # 1. 读取数据 (注意处理字符编码)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"原始数据规模 -> 训练集: {train_df.shape}, 测试集: {test_df.shape}")

    # 2. 分离目标变量，防止数据泄露
    target_col = '血糖'
    y_train = train_df[target_col].copy()

    # 3. 提取特征矩阵并合并，方便统一清洗
    # 给测试集加一个空的血糖列占位，方便对齐
    train_features = train_df.drop(columns=[target_col])
    test_features = test_df.copy()

    # 打上标签，方便一会拆分回来
    train_features['is_train'] = 1
    test_features['is_train'] = 0

    # 合并特征
    all_features = pd.concat([train_features, test_features], ignore_index=True)

    # 4. 删除冗余列和高缺失列
    cols_to_drop = [
        'id', '体检日期',
        '乙肝e抗体', '乙肝e抗原', '乙肝核心抗体', '乙肝表面抗体', '乙肝表面抗原'  # 缺失率超75%
    ]
    # 只删除确实存在于DataFrame中的列
    cols_to_drop = [c for c in cols_to_drop if c in all_features.columns]
    all_features.drop(columns=cols_to_drop, inplace=True)
    print(f"✂️ 已剔除高缺失及冗余特征：{cols_to_drop}")

    # 5. 类别特征编码：性别
    if '性别' in all_features.columns:
        # 清理可能存在的脏数据（如带空格的男女，或者未知）
        all_features['性别'] = all_features['性别'].str.strip()
        all_features['性别'] = all_features['性别'].map({'男': 1, '女': 0})
        # 如果有少量缺失的性别，用众数填充
        all_features['性别'].fillna(all_features['性别'].mode()[0], inplace=True)
        all_features['性别'] = all_features['性别'].astype(int)

    # 强制将所有列转换为数值型，无法转换的变为 NaN
    cols_to_convert = [c for c in all_features.columns if c != 'is_train']
    for col in cols_to_convert:
        all_features[col] = pd.to_numeric(all_features[col], errors='coerce')

    # 6. 异常值处理 (盖帽法 Winsorization)
    print("🛡️ 正在进行极端异常值截断 (1% - 99%)...")
    numeric_cols = [col for col in all_features.columns if col not in ['性别', 'is_train']]
    for col in numeric_cols:
        # 计算 1% 和 99% 分位数
        lower_bound = all_features[col].quantile(0.01)
        upper_bound = all_features[col].quantile(0.99)
        # 将超出边界的值截断
        all_features[col] = np.clip(all_features[col], lower_bound, upper_bound)

    # 7. 高阶缺失值插补 (KNN Imputer)
    print("🧬 正在利用 KNN 算法进行多变量缺失值插补 (这可能需要几秒钟)...")
    # 提取需要插补的特征列 (不包括 is_train)
    features_to_impute = all_features.drop(columns=['is_train'])
    feature_names = features_to_impute.columns

    # 初始化KNN插补器，n_neighbors=5 通常是一个稳健的默认值
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    imputed_array = imputer.fit_transform(features_to_impute)

    # 将插补后的数组转回 DataFrame
    imputed_df = pd.DataFrame(imputed_array, columns=feature_names)
    imputed_df['is_train'] = all_features['is_train'].values

    # 8. 拆分回训练集和测试集
    clean_train_features = imputed_df[imputed_df['is_train'] == 1].drop(columns=['is_train']).reset_index(drop=True)
    clean_test_features = imputed_df[imputed_df['is_train'] == 0].drop(columns=['is_train']).reset_index(drop=True)

    # 将目标变量拼回训练集
    clean_train_df = pd.concat([clean_train_features, y_train.reset_index(drop=True)], axis=1)

    # 剔除由于 target (血糖) 本身为 NaN 的训练样本（如果有的话，监督学习不能没有标签）
    clean_train_df.dropna(subset=[target_col], inplace=True)

    clean_test_df = clean_test_features.copy()

    # 9. 导出结果
    clean_train_df.to_csv(out_train_path, index=False)
    clean_test_df.to_csv(out_test_path, index=False)

    print("✅ 处理完成！")
    print(f"输出数据规模 -> 训练集: {clean_train_df.shape}, 测试集: {clean_test_df.shape}")
    print(f"数据已分别保存至: '{out_train_path}' 和 '{out_test_path}' (UTF-8编码)")


# ================= 运行调用 =================
if __name__ == '__main__':
    TRAIN_INPUT = 'with_blood.csv'
    TEST_INPUT = 'within_blood.csv'

    TRAIN_OUTPUT = 'cleaned_with_blood.csv'
    TEST_OUTPUT = 'cleaned_within_blood.csv'

    preprocess_diabetes_data(TRAIN_INPUT, TEST_INPUT, TRAIN_OUTPUT, TEST_OUTPUT)