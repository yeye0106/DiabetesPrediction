import pandas as pd
# pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
# pd.set_option('display.width', None)


def missing_report(df):
    n = len(df)
    missing_info = []
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_pct = missing_count / n * 100
            missing_info.append((col, missing_count, missing_pct))
    # 按缺失数量降序排序
    missing_info.sort(key=lambda x: x[1], reverse=True)
    for col, count, pct in missing_info:
        print(f"{col}: {count} 缺失值, 占比 {pct:.2f}%")

df = pd.read_csv('X_test_preprocessed.csv')
print(df.head(2))
columns1 = df.columns
print(columns1)
print(df.shape)
missing_report(df)


df = pd.read_csv('X_train_preprocessed.csv')
print(df.head(2))
columns2 = df.columns
print(columns2)
print(df.shape)
missing_report(df)
