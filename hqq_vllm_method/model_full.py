import torch
from transformers import AutoModelForCausalLM


device = 'cuda:1'
model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=device,
)
model.eval()

# list all the name and module
for name, module in model.named_modules():
    print(name)
