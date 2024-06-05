import json
import os
from typing import List, Dict, Optional, Any

import torch
import jieba
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from openai import OpenAI
from rouge_chinese import Rouge
# from udicOpenData.stopwords import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 預先加載 model 和 tokenizer
FINE_TUNED_MODEL_NAME = 'model/taide_markdown_v3-full'
ORIGINAL_MODEL_NAME = 'model/Llama3-TAIDE-LX-8B-Chat-Alpha1'
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_NAME)
fine_tuned_model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_MODEL_NAME, load_in_8bit=True, device_map='auto')
original_tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_NAME)
original_model = AutoModelForCausalLM.from_pretrained(ORIGINAL_MODEL_NAME, load_in_8bit=True, device_map='auto')

load_dotenv(".env")
client = OpenAI(api_key=os.getenv('openai_api_key'))

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

def get_taide_prompt(document: str, question: str, version: str = 'llama3') -> str:
    """產生TAIDE針對農業病蟲害問題的prompt。"""
    if version == "llama2":
        system_prompt = '你是一位農業病蟲害防治專家，你將看到一份markdown格式的表格，你的任務是根據此表格尋找答案並回答，回答時只需要輸出答案，不需輸出其他資訊。'
        user_message = f"問題：\n{question}\n\n表格：\n{document}\n\n"
        return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"   
     
    elif version == "llama3":
        system_prompt = "你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。"
        user_message = f"你是一位農業病蟲害防治專家，你將看到一份markdown格式的表格，你的任務是根據此表格尋找答案並回答，回答時只需要輸出答案，不需輸出其他資訊。\n問題:\n{question}\n\n表格:\n{document}"
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
    else:
        user_message = f"你是一位農業病蟲害防治專家，你將看到一份markdown格式的表格，你的任務是根據此表格尋找答案並回答，回答時只需要輸出答案，不需輸出其他資訊。\n問題:\n{question}\n\n表格:\n{document}"
        return user_message

def get_gpt_prompt(question: str, reference_article: str, answer1: str, answer2: str, gpt_answer: str) -> str:
    """根據問題和文章產生GPT的prompt。"""
    system_prompt = "你是一位農業病蟲害防治專家，你將看到一個問題、一份markdown格式的表格做為參考文章，以及三個針對這個問題的回答。你的任務是根據問題和表格內容，判斷回答1及回答2哪個更接近回答3。\n"+"注意：回覆時只需輸出'回答1'、'回答2'、'一樣好'\n\n"
    user_message = f"問題：\n{question}\n\n表格：\n{reference_article}\n\n回答1：\n{answer1}\n\n回答2：\n{answer2}\n\n回答3：\n{gpt_answer}"
    return f"{system_prompt}\n{user_message}"

def get_taide_reply(tokenizer, model, prompt: str) -> str:
    """使用TAIDE模型生成回答。"""
    model.eval()
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    generation_config = GenerationConfig(
        max_new_tokens=300,
        top_k=50,
        num_beams=2,
        num_return_sequences=1,
        early_stopping=True,
    )
    outputs = model.generate(**inputs, generation_config=generation_config)
    reply = tokenizer.batch_decode(outputs[:, inputs.input_ids.size(1) :], skip_special_tokens=True)[0]
    return reply

def get_gpt_reply(content: str) -> str:
    """使用GPT模型生成回答"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "user",
                "content": f"{content}",
            }
        ],
        temperature=0,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    resp = response.choices[0].message.content
    return resp

def parse_gpt_response(response: str) -> Optional[str]:
    """解析GPT模型的回答並提取判斷結果。"""
    try:
        if "一樣好" in response:
            return 3
        elif "回答1" in response:
            return 1
        elif "回答2" in response:
            return 2
        else:
            return response
    except Exception as e:
        print("解析回答時出現錯誤:", e)
        return None

def get_gpt_decision(question: str, document: str, answer1: str, answer2: str, gpt_answer: str, is_original_first: bool = True) -> Optional[str]:
    """根據GPT的回答決定較好的選擇。"""
    comparison_prompt = get_gpt_prompt(question, document, answer1, answer2, gpt_answer)
    gpt_judgement = get_gpt_reply(comparison_prompt)
    best_answer = parse_gpt_response(gpt_judgement)

    if best_answer is None:
        return None
    elif best_answer == 1:
        return "raw" if is_original_first else "tuned"
    elif best_answer == 2:
        return "tuned" if is_original_first else "raw"
    elif best_answer == 3:
        return "equal"
    else:
        return best_answer

def tokenized_text(text: str) -> List[str]:
    """將文本轉換為分詞後的單詞列表。"""
    # word_list = list(rmsw(text, flag=False))
    word_list = list(jieba.cut(text))
    return ' '.join(word_list)    

def calculate_rouge_score(hypothesis: str, reference: str) -> Dict[str, float]:
    """計算並返回Rouge-1和Rouge-L分數。"""
    rouge = Rouge()
    scores = rouge.get_scores(tokenized_text(hypothesis), tokenized_text(reference))[0]
    return {
        'rouge-1': scores['rouge-1']['f'],
        'rouge-L': scores['rouge-l']['f']
    }
        
if __name__ == "__main__":
    input_filepath = "dataset/generate/generated_data_60_new.json"  # 輸入檔案名稱
    output_filepath = "dataset/compare/test_compare_60_v4.json"  # 輸出檔案名稱
    
    data = load_json_data(input_filepath)
    
    results = []
    half_data_len = len(data) // 2

    for index, item in enumerate(tqdm(data, desc="處理進度")):
        is_original_first = index < half_data_len  # 資料後半段調換答案擺放順序
        
        document = item['document']
        question = item['question']
        answer = item['answer']
        
        prompt_original = get_taide_prompt(document, question, version='llama3')
        original_answer = get_taide_reply(original_tokenizer, original_model, prompt_original)
        prompt_tuned = get_taide_prompt(document, question, version='llama3')
        fine_tuned_answer = get_taide_reply(fine_tuned_tokenizer, fine_tuned_model, prompt_tuned)
          
        # if is_original_first:  # 資料後半段調換答案擺放順序
        #     best_answer = get_gpt_decision(question, document, original_answer, fine_tuned_answer, is_original_first)
        # else:
        #     best_answer = get_gpt_decision(question, document, fine_tuned_answer, original_answer, is_original_first)
        
        rouge_scores = {
            'ground_truth_vs_llama3': calculate_rouge_score(answer, fine_tuned_answer),
            'ground_truth_vs_llama2': calculate_rouge_score(answer, original_answer)
        }
        
        data_entry = {
            'id': index,
            'document': document,
            'question': question,
            # 'best_answer': best_answer,
            'answers': {
                'ground_truth': answer,
                'tuned': fine_tuned_answer,
                'raw': original_answer
            },
            'rouge_scores': rouge_scores
        }
        results.append(data_entry)

    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)
        print(f"結果已成功儲存至 {output_filepath}。")
