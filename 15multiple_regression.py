import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def perform_regression_analysis():
    # 读取数据
    file_path = r'E:\中科院\情绪多样性、跨模态跨来源情绪一致性、负向结论（过于激动不好）\手工编码\一致性或多样性回归\S13-S15回归分析_diversity_processed.xlsx'  #!!改改改！！！！！！！！！！！！！！！！！！！！！！
    df = pd.read_excel(file_path)
    
    # 创建保存热力图的目录
    heatmap_dir = r'E:\中科院\情绪多样性、跨模态跨来源情绪一致性、负向结论（过于激动不好）\手工编码\一致性或多样性回归\diversity_heatmaps'   #!!改改改！！！！！！！！！！！！！！！！！！！！！！
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # 定义内容级和情感级变量
    # content_vars = ['topic_consistency(bertopic)', 'topic_consistency(LDA)', 'semantics_consistency(bert)']      #!!改改改！！！！！！！！！！！！！！！！！！！！！！
    # emotion_vars = ['emotion_consistency(NRC8)', 'sentiment_consistency(bert3)', 'sentiment_consistency(NRC2)']
    content_vars = ['topic_diversity(bertopic)', 'topic_diversity(LDA)', 'semantics_diversity(bert)']
    emotion_vars = ['emotion_diversity(NRC8)', 'sentiment_diversity(bert3)', 'sentiment_diversity(NRC2)']
    
    # 创建一个Excel写入器
    output_path = file_path.replace('.xlsx', '_regression_results_all_combinations.xlsx')
    writer = pd.ExcelWriter(output_path, engine='openpyxl')
    
    # 存储所有组合的结果
    all_results = []
    
    # 遍历所有可能的组合
    for content_var in content_vars:
        for emotion_var in emotion_vars:
            print(f"\n=== 分析组合：{content_var} + {emotion_var} ===")
            
            # 准备自变量
            X = df[[content_var, emotion_var]]
            y_success = df['success']
            y_valuation = df['Financial_valuation_ln_standardized']
            
            # # 标准化自变量和Financial valuation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            # y_valuation_scaled = scaler.fit_transform(y_valuation.values.reshape(-1, 1)).flatten()
            
            # 添加常数项
            X_with_const = sm.add_constant(X_scaled)
            
            # 对success进行回归分析
            model_success = sm.OLS(y_success, X_with_const).fit()
            print("\n因变量：success")
            print(model_success.summary())
            
            # 对标准化后的Financial valuation进行回归分析
            model_valuation = sm.OLS(y_valuation, X_with_const).fit()
            print("\n因变量：Financial valuation（对数和标准化后）")
            print(model_valuation.summary())
            
            # 计算VIF值
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X_scaled.columns
            vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                              for i in range(1, X_with_const.shape[1])]
            print("\n多重共线性检验（VIF值）：")
            print(vif_data)
            
            # 存储结果
            combination_results = {
                '内容级变量': content_var,
                '情感级变量': emotion_var,
                'success_R方': model_success.rsquared,
                'success_adjusted_R方': model_success.rsquared_adj,
                'success_F统计量': model_success.fvalue,
                'success_F统计量p值': model_success.f_pvalue,
                f'success_{content_var}_系数': model_success.params[content_var],
                f'success_{content_var}_p值': model_success.pvalues[content_var],
                f'success_{emotion_var}_系数': model_success.params[emotion_var],
                f'success_{emotion_var}_p值': model_success.pvalues[emotion_var],
                'valuation_R方': model_valuation.rsquared,
                'valuation_adjusted_R方': model_valuation.rsquared_adj,
                'valuation_F统计量': model_valuation.fvalue,
                'valuation_F统计量p值': model_valuation.f_pvalue,
                f'valuation_{content_var}_系数': model_valuation.params[content_var],
                f'valuation_{content_var}_p值': model_valuation.pvalues[content_var],
                f'valuation_{emotion_var}_系数': model_valuation.params[emotion_var],
                f'valuation_{emotion_var}_p值': model_valuation.pvalues[emotion_var]
            }
            all_results.append(combination_results)
            
            # 生成相关性热力图
            plt.figure(figsize=(10, 8))
            correlation_data = pd.DataFrame({
                content_var: X_scaled[content_var],
                emotion_var: X_scaled[emotion_var],
                'success': y_success,
                'Financial valuation (ln_std)': y_valuation
            })
            correlation_matrix = correlation_data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title(f'相关性热力图 ({content_var} + {emotion_var})')
            plt.tight_layout()
            
            # 保存热力图
            heatmap_filename = os.path.basename(file_path).replace('.xlsx', f'_heatmap_{content_var}_{emotion_var}.png')
            plt.savefig(os.path.join(heatmap_dir, heatmap_filename))
            plt.close()
    
    # 将所有结果转换为DataFrame并保存
    results_df = pd.DataFrame(all_results)
    results_df.to_excel(output_path, index=False)
    print(f"\n所有组合的回归分析结果已保存至：{output_path}")

if __name__ == "__main__":
    perform_regression_analysis() 