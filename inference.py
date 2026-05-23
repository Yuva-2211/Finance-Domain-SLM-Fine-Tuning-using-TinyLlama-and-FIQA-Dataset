from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Base model

base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Path to your trained LoRA adapter

adapter_path = "./finance_slm"

# Load tokenizer

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base model

base_model = AutoModelForCausalLM.from_pretrained(
base_model_name,
torch_dtype=torch.float16,
device_map="auto"
)

# Load LoRA adapter

model = PeftModel.from_pretrained(
base_model,
adapter_path
)

# Inference function

def ask_finance_question(question):

```
prompt = f"""
```

### Question:

{question}

### Answer:

"""

```
inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=120,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty=1.2,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

return response
```

# Interactive loop

while True:

```
question = input("\nEnter your finance question (or type 'exit'): ")

if question.lower() == "exit":
    break

answer = ask_finance_question(question)

print("\nGenerated Answer:\n")
print(answer)
```
