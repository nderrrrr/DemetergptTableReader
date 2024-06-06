import torch
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

# qlora configuration
bnb_4bit_compute_dtype = "bfloat16"
lora_r = 8
lora_alpha = lora_r * 2
lora_dropout = 0.1

# training configuration
model_path = "model/Llama3-TAIDE-LX-8B-Chat-Alpha1" # 原始模型
new_model_name = "taide_markdown_v5" # 訓練好的模型
train_dataset_path = "dataset/train/agri_llama3_train_v2.json"
test_dataset_path = "dataset/train/agri_llama3_dev_v2.json"
output_dir = "model/results" # tensorboard結果
instruction_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
fp16 = False
bf16 = True

num_train_epochs = 1
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
gradient_accumulation_steps = 4

max_grad_norm = 0.3
learning_rate = 5e-5
weight_decay = 0.001
warmup_ratio = 0.03
group_by_length = True

max_seq_length = 4096
max_steps = -1
logging_steps = 25

device_map = "auto"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = "[PAD]" # use when tokenizer.pad_token is none or eos_token
tokenizer.add_eos_token = True # 訓練資料不要加eos_token "</s>"
tokenizer.padding_side = "right"

# load model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
)
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
model.config.use_cache = False
model.config.pretraining_tp = 1

# load dataset
dataset = load_dataset('json',data_files={
    "train":train_dataset_path,
    "test":test_dataset_path
})


# trainer arguments
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type="cosine",
    report_to="tensorboard"
)

collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    dataset_text_field="text",
    data_collator=collator,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

trainer.train()
trainer.model.save_pretrained(new_model_name) # 儲存lora參數

torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model_name)
model = model.merge_and_unload() # 將lora與原始模型融合

model.save_pretrained(new_model_name+"-full") # 儲存完整模型
tokenizer.save_pretrained(new_model_name+"-full")
