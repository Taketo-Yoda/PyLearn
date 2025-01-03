import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./LLM", device_map="auto", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("./LLM")

input = tokenizer("東京は日本の", return_tensors="pt").to(model.device)

# Page3
print("----- Page3 -----")
tokens = model.generate(**input, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.eos_token_id)

print(tokenizer.decode(tokens[0][-1]))
print("-----------------")

# Page4
print("----- Page4 -----")
out = model.generate(
    **input,
    max_new_tokens=1,
    return_dict_in_generate=True,
    output_scores=True,
    pad_token_id=tokenizer.eos_token_id)

print(out.scores[0].shape)

top5 = torch.topk(out.scores[0][0], 5)
for i in range(5) :
    print(i+1, tokenizer.decode(top5.indices[i]), top5.values[i].item())
print("-----------------")

# Page6
print("----- Page6 -----")
input6 = tokenizer("日本の首都はどこですか？", return_tensors="pt").to(model.device)
tokens6 = model.generate(**input6, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(tokens6[0], skip_special_tokens=True))
print("-----------------")

# Page7
print("----- Page7 -----")
input7 = tokenizer(
    "今日は天気がよいですね\n" +
    "そうですね\n" +
    "どこかへ行きましょうか。",
    return_tensors="pt").to(model.device)
tokens7 = model.generate(**input7, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(tokens7[0], skip_special_tokens=True))
print("-----------------")
