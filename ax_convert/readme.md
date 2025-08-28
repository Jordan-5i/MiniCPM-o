




scp必要的运行文件到650的板子上

```
# llm用到的embeddings矩阵
scp ./embed_tokens.pth ./run_axmodel.py  root@10.126.29.158:/root/wangjian/minicpm-v-4

# 编译相关的axmodel
scp -r output_siglip/siglip.axmodel output_resampler/resampler.axmodel minicpm-v-4_axmodel root@10.126.29.158:/root/wangjian/minicpm-v-4

# 示例图片
scp /data/wangjian/project/MiniCPM-o/assets/minicpmo2_6/show_demo.jpg root@10.126.29.158:/root/wangjian/minicpm-v-4

# huggingface的repo
scp /data/wangjian/project/hf_cache/openbmb/MiniCPM-V-4/{*.py,*.json} root@10.126.29.158:/root/wangjian/hf_cache/MiniCPM-V-4
```