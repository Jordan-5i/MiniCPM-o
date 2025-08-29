# 模型转化
PS：`transformers==4.51.0`，其他版本可能运行时会有问题。

## 导出minicpm-v中siglip和resampler模型（PyTorch->ONNX）
- siglip 用于图像特征处理
- resampler 用于将silip的输出feature压缩成固定数量的feature，resampler本身带有weight

下载huggingface的minicpm-v-4到本地：https://huggingface.co/openbmb/MiniCPM-V-4

运行minicpm-v-4的repo
```bash
cd ax_convert
python run.py
```

运行导出onnx脚本：
```bash
cd ax_convert
python export_onnx.py 
```
会在ax_convert目录上生成siglip.onnx，resampler.onnx，embed_tokens.pth。用onnxslim优化一下模型：
```bash
onnxslim siglip.onnx siglip-slim.onnx

onnxslim resampler.onnx resampler-slim.onnx
```
> embed_tokens.pth 是llm中词向量权重矩阵

可选：验证onnx是否正确
运行脚本：
```bash
python run_siglip_onnx.py
```

## 编译模型（ONNX->AXMODEL）
以提供必要的编译配置文件：`config_resampler.json`，`config_siglip.json`

生成编译模型需要的量化数据集，运行脚本：   
```bash
python prepare_calib.py
```
会在当前目录上生成calib_resampler_data，calib_siglip_data两个目录，里面的siglip.tar和resampler.tar文件，就是模型量化数据集

pulsar2 编译模型
```bash
# siglip 模型
pulsar2 build --input siglip-slim.onnx --config config_siglip.json --output_dir output_siglip --debug.dump_frontend_graph --output_name siglip.axmodel

# resampler 模型
pulsar2 build --input resampler-slim.onnx --config config_resampler.json --output_dir output_resampler --debug.dump_frontend_graph --output_name resampler.axmodel

# llm 模型
pulsar2 llm_build --input_path /data/wangjian/project/hf_cache/openbmb/MiniCPM-V-4  --output_path minicpm-v-4_axmodel/ --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 320 --parallel 32 --chip AX650
```

## AX650端板推理
scp必要的运行文件到650的板子上

```bash
# llm用到的embeddings矩阵，推理脚本run_axmodel.py
scp ./embed_tokens.pth ./run_axmodel.py  root@10.126.29.158:/root/wangjian/minicpm-v-4

# 编译相关的axmodel
scp -r output_siglip/siglip.axmodel output_resampler/resampler.axmodel minicpm-v-4_axmodel root@10.126.29.158:/root/wangjian/minicpm-v-4

# 示例图片
scp /data/wangjian/project/MiniCPM-o/assets/minicpmo2_6/show_demo.jpg root@10.126.29.158:/root/wangjian/minicpm-v-4

# huggingface的repo
scp /data/wangjian/project/hf_cache/openbmb/MiniCPM-V-4/{*.py,*.json} root@10.126.29.158:/root/wangjian/hf_cache/MiniCPM-V-4
```

运行推理脚本：
```bash
python run_axmodel.py
```
