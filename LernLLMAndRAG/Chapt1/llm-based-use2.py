import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "./LLM"

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Page14
print("----- Page14 -----")
generater = pipeline("text-generation", model=model, tokenizer=tokenizer)

outs = generater("東京は日本の", max_new_tokens=30)
print(outs[0])
print("------------------")
