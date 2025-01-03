import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./LLM", device_map="auto", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("./LLM")

input = tokenizer("東京は日本の", return_tensors="pt").to(model.device)
tokens = model.generate(**input, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.eos_token_id)

print(tokenizer.decode(tokens[0][-1]))