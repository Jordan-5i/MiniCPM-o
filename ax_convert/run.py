import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor

torch.manual_seed(100)

model = AutoModel.from_pretrained('/data/wangjian/project/hf_cache/openbmb/MiniCPM-V-4', trust_remote_code=True, # or openbmb/MiniCPM-o-2_6
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
processor = AutoProcessor.from_pretrained('/data/wangjian/project/hf_cache/openbmb/MiniCPM-V-4', trust_remote_code=True)
model.config.slice_mode = False  # 不对图像做切分操作
processor.image_processor.slice_mode = False
model.processor = processor
model = model.eval().cuda(1)
tokenizer = AutoTokenizer.from_pretrained('/data/wangjian/project/hf_cache/openbmb/MiniCPM-V-4', trust_remote_code=True) # or openbmb/MiniCPM-o-2_6

image = Image.open('../assets/minicpmo2_6/show_demo.jpg').convert('RGB').resize((448, 448))

# First round chat 
question = "What is the landform in the picture?"
msgs = [{'role': 'user', 'content': [image, question]}]

answer = model.chat(
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)

# Second round chat, pass history context of multi-turn conversation
msgs.append({"role": "assistant", "content": [answer]})
msgs.append({"role": "user", "content": ["What should I pay attention to when traveling here?"]})

answer = model.chat(
    msgs=msgs,
    tokenizer=tokenizer
)
print('---')
print(answer)

"""
The landform in the picture is karst topography, characterized by steep limestone hills and cliffs. This type of landscape typically forms due to the dissolution of soluble rocks like limestone over time, creating unique shapes that are often seen in regions with significant geological activity.
---
When traveling to areas with karst topography, it's important to be mindful of the terrain. The landscape can present challenges such as uneven ground and potential rockfalls. Additionally, always respect local ecosystems by not disturbing flora or fauna. If you plan to explore further into these hills, consider bringing sturdy footwear for hiking.
"""