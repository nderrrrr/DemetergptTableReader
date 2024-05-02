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
FINE_TUNED_MODEL_NAME = 'taide_markdown_v2-full'
ORIGINAL_MODEL_NAME = '/user_data/llama2_lora_drcd/taide_model/b.11.0.0'
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_NAME)
fine_tuned_model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_MODEL_NAME, load_in_8bit=True, device_map='auto')
original_tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_NAME)
original_model = AutoModelForCausalLM.from_pretrained(ORIGINAL_MODEL_NAME, load_in_8bit=True, device_map='auto')

def load_json_data(filename: str) -> List[Dict[str, Any]]:
    """從JSON文件中加載數據。"""
    try:
        with open(filename, "r", encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print("錯誤：未找到檔案。")
    except json.JSONDecodeError:
        print("錯誤：檔案格式不正確。")
    except Exception as e:
        print(f"發生了意外錯誤：{e}")
    return []

def get_taide_prompt(document: str, question: str) -> str:
    """產生TAIDE針對農業病蟲害問題的prompt。"""
    system_prompt = '<<SYS>>\n你是一位農業病蟲害防治專家，你將看到一份markdown格式的表格，你的任務是根據此表格尋找答案並回答，回答時只需要輸出答案，不需輸出其他資訊\n<</SYS>>\n\n'
    user_message = f"問題：\n{question}\n\n表格：\n{document}\n\n"
    return f"[INST] {system_prompt}\n{user_message} [/INST]"

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
        return "original" if is_original_first else "fine_tuned"
    elif best_answer == 2:
        return "fine_tuned" if is_original_first else "original"
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
    load_dotenv(".env")
    client = OpenAI(api_key=os.getenv('openai_api_key'))
    data = load_json_data('generated_data_60_new.json')  # 待比較的數據
    print(f"總共加載了 {len(data)} 筆數據。")

    output_filename = "train_compare_60_v2.json"  # 輸出檔案名稱
    results: List[Dict[str, Any]] = []
    half_data_len = len(data) // 2

    for index, item in enumerate(tqdm(data, desc="處理進度")):
        is_original_first = index < half_data_len  # 資料後半段調換答案擺放順序
        
        prompt = get_taide_prompt(item['document'], item['question'])
        original_answer = get_taide_reply(original_tokenizer, original_model, prompt)
        fine_tuned_answer = get_taide_reply(fine_tuned_tokenizer, fine_tuned_model, prompt)
          
        if is_original_first:  # 資料後半段調換答案擺放順序
            best_answer = get_gpt_decision(item['question'], item['document'], original_answer, fine_tuned_answer, is_original_first)
        else:
            best_answer = get_gpt_decision(item['question'], item['document'], fine_tuned_answer, original_answer, is_original_first)
        
        rouge_scores = {
            'gpt_vs_fine_tuned': calculate_rouge_score(item['answer'], fine_tuned_answer),
            'gpt_vs_original': calculate_rouge_score(item['answer'], original_answer)
        }
        
        data_entry = {
            'id': index,
            'document': item['document'],
            'question': item['question'],
            'best_answer': best_answer,
            'answers': {
                'gpt': item['answer'],
                'fine_tuned': fine_tuned_answer,
                'original': original_answer
            },
            'rouge_scores': rouge_scores
        }
        results.append(data_entry)

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)
        print(f"結果已成功儲存至 {output_filename}。")
