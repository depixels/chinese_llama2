from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass


device = "cuda" if torch.cuda.is_available() else 'cpu'
pretrained_model_name_or_path = 'hyz/llama2-7b-chat-hf'
model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        device_map = device,
        torch_dtype = torch.float16,
        load_in_8bit = True,
        trust_remote_code = True,
    )
tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code = True
    )
print(model)
print(tokenizer)
tokenizer.pad_token = tokenizer.eos_token
prompt = '<s><Human>: 你好，帮我介绍一下北京。</s><Assistant>:'
inputs = tokenizer(prompt, return_tensors='pt')
print(inputs)
print(type(inputs))
outptus = model.generate(inputs)