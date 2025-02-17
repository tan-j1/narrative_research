import pandas as pd
import os
import re

def extract_investment_details(txt_content):
    """
    从文本中提取投资金额（$或dollar或纯数字）和百分比
    在%符号前后的文本范围内搜索金额
    支持识别数字单位 (thousand, million)
    只匹配较大数字(>=1000)避免误匹配
    """
    def convert_to_number(amount_str):
        amount_str = amount_str.lower().replace(',', '')
        if 'thousand' in amount_str:
            return int(float(amount_str.replace('thousand', '').strip()) * 1000)
        elif 'million' in amount_str:
            return int(float(amount_str.replace('million', '').strip()) * 1000000)
        return int(amount_str)

    def find_investment_in_text(text):
        # 尝试匹配$格式
        dollar_match = re.search(r'\$(\d+(?:,\d{3})*(?:\s*(?:thousand|million)?)?)', text, re.IGNORECASE)
        if not dollar_match:
            # 尝试匹配dollar格式
            dollar_match = re.search(r'(\d+(?:,\d{3})*(?:\s*(?:thousand|million)?)?)\s*dollars?', text, re.IGNORECASE)
        if not dollar_match:
            # 尝试匹配纯数字格式（至少包含000或带单位）
            number_match = re.search(r'(\d+,\d{3}|\d+\s*(?:thousand|million))', text, re.IGNORECASE)
            if number_match:
                dollar_match = number_match
        
        if dollar_match:
            amount_str = dollar_match.group(1)
            try:
                amount = convert_to_number(amount_str)
                if amount >= 1000:  # 只返回>=1000的数值
                    return amount
            except:
                pass
        return None

    # 找到百分比位置
    percent_match = re.search(r'(\d+)%', txt_content)
    expected_stake = None
    expected_investment = None
    
    if percent_match:
        expected_stake = int(percent_match.group(1))
        percent_pos = percent_match.start()
        
        # 在%号前后各300个字符范围内搜索
        search_range = 300
        start_pos = max(0, percent_pos - search_range)
        end_pos = min(len(txt_content), percent_pos + search_range)
        
        # 提取%号前后的文本
        before_text = txt_content[start_pos:percent_pos]
        after_text = txt_content[percent_pos:end_pos]
        
        # 先在%号前的文本中搜索
        expected_investment = find_investment_in_text(before_text)
        
        # 如果在前面没找到，再在后面搜索
        if expected_investment is None:
            expected_investment = find_investment_in_text(after_text)
            
        # 如果在范围内都没找到，在整个文本中搜索
        if expected_investment is None:
            expected_investment = find_investment_in_text(txt_content)
    else:
        # 如果没有找到%，在整个文本中搜索
        expected_investment = find_investment_in_text(txt_content)
    
    return expected_investment, expected_stake

def update_analysis_table():
    # 读取最终分析表(3)
    excel_path = r'E:\中科院\情绪多样性、跨模态跨来源情绪一致性、负向结论（过于激动不好）\手工编码\最终分析表(4).xlsx'
    df = pd.read_excel(excel_path)
    

    # txt文件夹路径
    txt_folder = r'E:\中科院\shark tank视频下载\raw_data season13_3'
    
    # 遍历DataFrame的每一行
    for index, row in df.iterrows():
        file_name = row['file_name']
        if pd.isna(file_name):
            continue
            
        # 构建完整的txt文件路径
        txt_path = os.path.join(txt_folder, file_name + '.txt')
        
        if not os.path.exists(txt_path):
            print(f"找不到文件: {txt_path}")
            continue
            
        try:
            # 读取txt文件内容
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取投资金额和股权比例
            investment, stake = extract_investment_details(content)
            
            # 更新DataFrame
            if investment is not None:
                df.at[index, 'expected_investment'] = investment
            if stake is not None:
                df.at[index, 'expected_stake'] = stake
                
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")
    
    # 保存更新后的Excel文件
    df.to_excel(excel_path, index=False)
    print("更新完成！")

if __name__ == "__main__":
    update_analysis_table()
