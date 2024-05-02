import json
import random
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split

def convert_to_llama_prompt(json_file: str) -> List[Dict[str, str]]:
    """將 JSON 檔案中的數據轉換成適合 LLaMA 模型訓練的格式。"""
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    random.shuffle(data)
    
    prompts = []
    for entry in data:
        text = f"[INST] <<SYS>>你是一位農業病蟲害防治專家，你將看到一份markdown格式的表格，你的任務是根據此表格尋找答案並回答，回答時只需要輸出答案，不需輸出其他資訊，\n<</SYS>>問題:\n{entry['question']}\n\n表格:\n{entry['document']}\n [/INST]{entry['answer']} </s>"
        prompts.append({"text": text})
    
    random.shuffle(prompts)
    return prompts

def length_filter(data: List[Dict[str, Any]], max_length: int) -> List[Dict[str, Any]]:
    """過濾掉超過指定長度的數據。"""
    return [entry for entry in data if len(entry['text']) <= max_length]

def split_data(data: List[Dict[str, Any]], test_size: float = 0.2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """將數據分割成訓練集和驗證集。"""
    train_data, dev_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, dev_data

def save_file(data: List[Dict[str, Any]], filename: str) -> None:
    """將數據保存到 JSON 檔案中。"""
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    input_file = 'dataset/generate/generated_data_600.json'  # 原始數據檔案的路徑
    train_file = 'dataset/train/markdown_taide_train.json'    # 保存訓練數據的檔案路徑
    dev_file = 'dataset/train/markdown_taide_dev.json'        # 保存開發數據的檔案路徑

    prompts = convert_to_llama_prompt(input_file)
    prompts = length_filter(prompts, 4096)
    train_data, dev_data = split_data(prompts)
    save_file(train_data, train_file)
    save_file(dev_data, dev_file)

    print("數據成功轉換並保存。")
