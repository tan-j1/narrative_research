import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from scipy.spatial.distance import jensenshannon

# 初始化BERT模型和分词器
def initialize_bert_model():
    print("初始化BERT模型和分词器...")
    model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'  # 可根据需要选择合适的模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    print("BERT模型和分词器初始化完成。")
    return tokenizer, model

# 使用BERT进行情感分类
def classify_sentiment(text, tokenizer, model, device):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    # 假设模型输出为5类情感，将其转换为三类
    # 例如：1-2类为负向，3类为中性，4-5类为正向
    # 根据具体模型调整
    sentiment_vector = np.zeros(3)
    if model.config.num_labels == 5:
        # 示例转换
        sentiment_vector[0] = probabilities[0] + probabilities[1]  # 负向
        sentiment_vector[1] = probabilities[2]  # 中性
        sentiment_vector[2] = probabilities[3] + probabilities[4]  # 正向
    elif model.config.num_labels == 3:
        sentiment_vector = probabilities  # 直接对应负、中、正
    else:
        # 根据模型输出自行调整
        raise ValueError("不支持的情感分类标签数。")
    
    return sentiment_vector

# 读取团队成员的内容
def read_team_content(base_paths, folder_name, team_members, tokenizer, model, device):
    print("读取团队成员内容...")
    members_content = {}
    folder_found = False
    
    # 在所有基础路径中查找对应的文件夹
    for base_path in base_paths:
        csv_dir = os.path.join(base_path, folder_name)
        if os.path.exists(csv_dir):
            folder_found = True
            for member_id in team_members:
                file_name = f'说话人_{member_id}.csv'
                file_path = os.path.join(csv_dir, file_name)
                
                if not os.path.exists(file_path):
                    print(f"错误：找不到说话人 {member_id} 的文件")
                    return None
                
                df = pd.read_csv(file_path)
                full_content = ' '.join(df['内容'].astype(str))  # 合并所有内容
                members_content[member_id] = full_content
            break  # 找到文件夹后就跳出循环
    
    if not folder_found:
        print(f"错误：在所有路径中都找不到文件夹 {folder_name}")
        return None
    
    print("团队成员内容读取完成。")
    return members_content

# 计算JSD（Jensen-Shannon Divergence）
def calculate_jsd(vec1, vec2):
    """计算两个分布之间的JSD"""
    # 确保向量是非负的
    vec1 = np.abs(vec1)
    vec2 = np.abs(vec2)
    
    # 处理零向量的情况
    if np.sum(vec1) == 0 or np.sum(vec2) == 0:
        return 1.0  # 如果有一个向量全为0，返回最大差异度1
    
    # 归一化为概率分布
    vec1 = vec1 / np.sum(vec1)
    vec2 = vec2 / np.sum(vec2)
    
    # 处理可能的数值问题
    vec1 = np.clip(vec1, 1e-10, 1)  # 避免出现0值
    vec2 = np.clip(vec2, 1e-10, 1)  # 避免出现0值
    
    # 重新归一化
    vec1 = vec1 / np.sum(vec1)
    vec2 = vec2 / np.sum(vec2)
    
    # 计算JSD
    jsd = jensenshannon(vec1, vec2)
    
    # 处理可能的nan值
    if np.isnan(jsd):
        return 1.0
        
    return jsd ** 2  # 返回平方后的JSD

# 计算情感一致性或多样性
def calculate_emotion_consistency(base_paths, folder_name, team_members, tokenizer, model, device, type="consistency"):
    if len(team_members) <= 1:  # 如果团队成员少于等于1，直接返回None，不做相似度计算
        print("团队成员数量少于等于1，无法计算。")
        return None, None  # 返回 None 而不是 0
    
    print(f"计算情感{type}...")
    
    members_content = read_team_content(base_paths, folder_name, team_members, tokenizer, model, device)
    
    if members_content is None:
        return None, None
    
    # 计算每个成员的情感向量
    emotion_vectors = {}
    for member_id, content in members_content.items():
        sentiment_vector = classify_sentiment(content, tokenizer, model, device)
        emotion_vectors[member_id] = sentiment_vector
    
    # 计算情感一致性或多样性
    consistency = {}
    member_ids = list(emotion_vectors.keys())
    for i in range(len(member_ids)):
        for j in range(i + 1, len(member_ids)):
            member_id = member_ids[i]
            other_id = member_ids[j]
            vector = emotion_vectors[member_id]
            other_vector = emotion_vectors[other_id]
            
            if type == "consistency":
                # 使用余弦相似度计算一致性
                vector = vector.reshape(1, -1)
                other_vector = other_vector.reshape(1, -1)
                similarity = cosine_similarity(vector, other_vector)[0][0]
            else:
                # 使用JSD计算多样性
                similarity = calculate_jsd(vector, other_vector)
            
            consistency[(member_id, other_id)] = similarity
    
    # 计算统计信息
    similarity_values = list(consistency.values())
    summary = {
        '团队成员数': len(team_members),
        '平均值': np.mean(similarity_values) if similarity_values else 0,
        '最低值': np.min(similarity_values) if similarity_values else 0,
        '最高值': np.max(similarity_values) if similarity_values else 0,
    }

    print("\n统计信息：")
    print(f"团队成员数: {summary['团队成员数']}")
    print(f"平均值: {summary['平均值']}")
    print(f"最低值: {summary['最低值']}")
    print(f"最高值: {summary['最高值']}")

    return consistency, summary

# 读取自我介绍说话人(2)文件，计算情感一致性并更新
def update_sentiment_consistency(input_file, output_file, base_paths, tokenizer, model, device, type="consistency"):
    print("开始读取并更新文件...")
    
    # 根据计算类型选择输出文件名
    if type == "diversity":
        output_file = output_file.replace(".xlsx", "_diversity.xlsx")
    else:
        output_file = output_file.replace(".xlsx", "_consistency.xlsx")
    
    # 如果输出文件已存在，读取它；否则读取输入文件
    try:
        df = pd.read_excel(output_file)
        print("读取现有输出文件成功")
    except FileNotFoundError:
        df = pd.read_excel(input_file)
        print("输出文件不存在，读取输入文件")
    
    # 新建列来存储情感一致性或多样性
    column_name = f'sentiment_{type}(bert3)'
    df[column_name] = None

    # 遍历每行数据
    for index, row in df.iterrows():
        print(f"\n处理第 {index+1} 行，文件名：{row['file_name']}")
        # 获取团队成员
        team_members = row['team']
        team_members = team_members.strip('[]').split(',')  # 去掉方括号并用逗号分隔
        team_members = [int(member.strip()) for member in team_members]  # 转换为整数
        
        # 计算情感一致性或多样性
        consistency_results, summary = calculate_emotion_consistency(base_paths, row['file_name'], team_members, tokenizer, model, device, type)
        
        # 如果存在情感一致性结果，取其中的平均相似度作为一致性值
        if summary:
            avg_similarity = summary['平均值']
            df.at[index, column_name] = avg_similarity
        else:
            df.at[index, column_name] = None  # 若没有结果，设置为None

    # 保存修改后的表格
    df.to_excel(output_file, index=False)
    print(f"结果已保存至 {output_file}")

# 主函数
def main():
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化BERT模型和分词器
    tokenizer, model = initialize_bert_model()
    model.to(device)
    model.eval()  # 设置为评估模式

    # 使用示例
    input_file = r'E:\中科院\情绪多样性、跨模态跨来源情绪一致性、负向结论（过于激动不好）\手工编码\S13-S15回归分析.xlsx'
    output_file = r'E:\中科院\情绪多样性、跨模态跨来源情绪一致性、负向结论（过于激动不好）\手工编码\S13-S15回归分析.xlsx'
    
    # 定义所有基础路径
    base_paths = [
        r'E:\中科院\shark tank视频下载\processed_data_season13',
        r'E:\中科院\shark tank视频下载\processed_data_season14',
        r'E:\中科院\shark tank视频下载\processed_data_season15'
    ]
    
    # 计算一致性
    update_sentiment_consistency(input_file, output_file, base_paths, tokenizer, model, device, type="diversity")
    
    # 计算多样性（取消注释以运行）
    # update_sentiment_consistency(input_file, output_file, base_paths, tokenizer, model, device, type="diversity")

if __name__ == "__main__":
    main()
