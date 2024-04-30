from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map="cuda:0")

messages = [{"role": "system", "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user."},{"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
inputs = inputs.to("cuda:0")

outputs = model.generate(inputs, max_new_tokens=40)
text = tokenizer.batch_decode(outputs)[0]
print(text)