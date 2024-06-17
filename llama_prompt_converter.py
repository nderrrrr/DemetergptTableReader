import os
import json
import random
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """從指定路徑讀取JSON文件，並處理潛在的例外情況。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        file_name = os.path.basename(file_path)
        print(f"文件 {file_name} 讀取成功，共讀取到 {len(data)} 筆數據。")
        return data
    except FileNotFoundError:
        print(f"Error: The file '{os.path.basename(file_path)}' does not exist.")
    except json.JSONDecodeError:
        print(f"Error: The file '{os.path.basename(file_path)}' contains invalid JSON.")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")
    return None

def get_taide_prompt(document: str, question: str, answer: str, version: str = 'llama3') -> str:
    """產生TAIDE針對農業病蟲害問題的prompt。"""
    if version == "llama2":
        system_prompt = '你是一位農業病蟲害防治專家，你將看到一份markdown格式的表格，你的任務是根據此表格尋找答案並回答，回答時只需要輸出答案，不需輸出其他資訊。'
        user_message = f"問題：\n{question}\n\n表格：\n{document}\n\n"
        return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]{answer}"   
     
    elif version == "llama3":
        system_prompt = "你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。"
        user_message = f"你是一位農業病蟲害防治專家，你將看到一份markdown格式的表格，你的任務是根據此表格尋找答案並回答，回答時只需要輸出答案，不需輸出其他資訊。\n問題:\n{question}\n\n表格:\n{document}"
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{answer}"
    
    else:
        user_message = f"你是一位農業病蟲害防治專家，你將看到一份markdown格式的表格，你的任務是根據此表格尋找答案並回答，回答時只需要輸出答案，不需輸出其他資訊。\n問題:\n{question}\n\n表格:\n{document}\n{answer}"
        return user_message

def length_filter(data: List[Dict[str, Any]], max_length: int = 4096) -> List[Dict[str, Any]]:
    """過濾掉超過指定長度的數據。"""
    return [entry for entry in data if len(entry['text']) <= max_length]

def split_data(data: List[Dict[str, Any]], test_size: float = 0.2):
    """將數據分割成訓練集和驗證集。"""
    train_data, dev_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, dev_data

def save_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """將數據保存到 JSON 檔案中。"""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"數據已保存到 {file_path}，總共 {len(data)} 筆資料。")

if __name__ == '__main__':
    input_file = 'dataset/generate/generated_data_600.json'  # 原始數據檔案的路徑
    
    train_file = 'dataset/train/agri_llama3_train_v2.json'    # 保存訓練數據的檔案路徑
    dev_file = 'dataset/train/agri_llama3_dev_v2.json'        # 保存驗證數據的檔案路徑

    converted_data = []
    raw_data = load_json_data(input_file)
    random.seed(42)
    random.shuffle(raw_data)
    
    for entry in raw_data:
        prompt = get_taide_prompt(entry['document'], entry['question'], entry['answer'], version='llama3')
        converted_data.append({
            'text': prompt,
        })
        
    prompts = length_filter(converted_data, 4000)
    train_data, dev_data = split_data(prompts, test_size=0.1)
    
    save_file(train_data, train_file)
    save_file(dev_data, dev_file)
