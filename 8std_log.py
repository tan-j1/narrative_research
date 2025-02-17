#在此之前需要在excel中手动计算一下Financial valuation，没有写代码处理是为了更好人工监测是否存在异常值或者异常录入！！！！！！！！！！！！！！！！！！！！！！！！！！
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# 读取Excel文件
file_path = r"E:\中科院\情绪多样性、跨模态跨来源情绪一致性、负向结论（过于激动不好）\手工编码\一致性或多样性回归\S13-S15回归分析_diversity.xlsx"
df = pd.read_excel(file_path)

# 处理Financial valuation中的异常值
financial_valuation = df['Financial valuation'].copy()
#financial_valuation = financial_valuation.replace([np.inf, -np.inf], np.nan) #极大值不用处理，感觉不如报错，因为正常情况下不可能出现极大值！！！！！！！！！！！！！！！！！！！
#financial_valuation = financial_valuation.fillna(financial_valuation.mean())

# 1. 对Financial valuation进行标准化
scaler = StandardScaler()
df['Financial_valuation_standardized'] = scaler.fit_transform(financial_valuation.values.reshape(-1, 1))

# 2. 对Financial valuation取ln (使用log1p处理零值)
df['Financial_valuation_ln'] = np.log1p(financial_valuation)

# 3. 对Financial valuation先取ln再标准化
ln_values = np.log1p(financial_valuation)
df['Financial_valuation_ln_standardized'] = scaler.fit_transform(ln_values.values.reshape(-1, 1))

# 保存结果到新的Excel文件
output_path = os.path.join(os.path.dirname(file_path), 'S13-S15回归分析_diversity_processed.xlsx')    
df.to_excel(output_path, index=False)

print("处理完成！新文件已保存至：", output_path)
print("\n数据统计信息：")
print(df[['Financial valuation', 'Financial_valuation_standardized', 'Financial_valuation_ln', 'Financial_valuation_ln_standardized']].describe())