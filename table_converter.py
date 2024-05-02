import json
import os
import re

from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

from typing import List, Dict, Any

load_dotenv(".env")
client = OpenAI(api_key=os.getenv('openai_api_key'))

def load_json(file_path: str):
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

def convert_to_markdown(table: Dict[str, Any]) -> str:
    """將給定的表格轉換為Markdown格式。"""
    header = table['header']
    rows = table['rows']
    markdown_table = "| " + " | ".join(header) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in rows:
        markdown_table += "| " + " | ".join(row) + " |\n"
    return markdown_table

def get_gpt_prompt(question: str, markdown_table: str, answers: List[str]) -> str:
    """獲得用於 GPT 請求的prompt。"""
    prompt = "你接下來會看到一組Table QA的資料，資料內容包含英文的問題、英文的表格、英文的答案。"\
            +"你的任務是把這筆資料轉成繁體中文，並將表格以Markdown的格式儲存。"\
            +"翻譯時需保留人名、地名、專有名詞等等，並確保翻譯後仍然能夠透過表格獲得答案。"\
            +"輸出時請以範例格式輸出"\
            +'[{"question": "translated_question", "markdown_table": "translated_table", "answers": ["translated_answers"]}]'\
            +f"問題：\n{question}\n\n"\
            +f"表格：\n{markdown_table}\n\n"\
            +f"答案：\n{answers}"

    return prompt

def get_gpt_reply(content: str) -> str:
    """使用指定的模型發送請求到 OpenAI 並獲取回覆。"""
    try:
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
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def parse_output(output_string: str) -> Dict[str, Any]:
    """使用正則表達式解析GPT回覆的輸出，尋找JSON格式的輸出。"""
    try:
        json_pattern = r'\[{"question": ".*?", "markdown_table": ".*?", "answers": \[".*?"\]}]'
        matches = re.search(json_pattern, output_string, re.DOTALL)
        if matches:
            parsed_output = json.loads(matches.group())
            if isinstance(parsed_output, list) and len(parsed_output) == 1:
                return parsed_output[0]
            else:
                raise ValueError("Output is not in the expected format.")
        else:
            raise ValueError("No valid JSON output found in the response.")
    except json.JSONDecodeError as e:
        print("JSON decoding error:", e)
        return {}
    except Exception as e:
        print("Error occurred:", e)
        return {}
    
def add_to_output(output_list, id, question, table, answers):
    output_list.append({
        'id': int(id),
        'question': question,
        'markdown_table': table,
        'answers': answers,
    })

def save_to_json(file_path: str, data: list):
    """將數據保存到指定的JSON文件。"""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"數據已保存到 {file_path}，總共 {len(data)} 筆資料。")


if __name__ == "__main__":
    file = '1000'
    
    file_path = f'dataset/wiki_table_question/WTQ_{file}.json'
    output_file_with_answer = f'dataset/wiki_table_question/WTQ_{file}_markdown_with_answer.json'
    output_file_all = f'dataset/wiki_table_question/WTQ_{file}_markdown_all.json'
    
    data = load_json(file_path)
    
    output_data_with_answer = []
    output_data_all = []
        
    for entry in tqdm(data, desc="資料轉換進度"):
        id = entry['id']
        question = entry['question']
        table = entry['table']
        answers = entry['answers']
        
        markdown_table = convert_to_markdown(table)
        prompt = get_gpt_prompt(question, markdown_table, answers)
        reply = get_gpt_reply(prompt)
        parsed_reply = parse_output(reply)
        
        if parsed_reply == {}:
            add_to_output(output_data_all, id, question, markdown_table, answers)
        else:
            question_zh = parsed_reply['question']
            markdown_table_zh = parsed_reply['markdown_table']
            answers_zh = parsed_reply['answers']
            
            answer_in_table_en = any(answer in markdown_table for answer in answers)
            answer_in_table_zh = any(answer in markdown_table_zh for answer in answers_zh)
            
            if answer_in_table_zh:
                add_to_output(output_data_all, id, question_zh, markdown_table_zh, answers_zh)
                add_to_output(output_data_with_answer, id, question_zh, markdown_table_zh, answers_zh)
            else:
                add_to_output(output_data_all, id, question_zh, markdown_table_zh, answers_zh)
                if answer_in_table_en:
                    add_to_output(output_data_with_answer, id, question, markdown_table, answers)
    
    save_to_json(output_file_with_answer, output_data_with_answer)
    save_to_json(output_file_all, output_data_all)
