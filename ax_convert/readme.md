# 模型转化
PS：`transformers==4.51.0`，其他版本可能运行时会有问题。

## 导出minicpm-v中siglip和resampler模型（PyTorch->ONNX）
- siglip 用于图像特征处理，类比vit
    - 输入shape=[1, 3, 448，448]
    - 输出shape=[1,1024,1152]
    - 1024是seq_len，1152是dim
- resampler 用于将siglip的输出feature压缩成固定数量的feature，resampler本身带有weight
    - 输入shape=[1,1024,1152]
    - 输出shape=[1,64,2560]

下载huggingface的[minicpm-v-4](https://huggingface.co/openbmb/MiniCPM-V-4)到本地，运行minicpm-v-4的repo
```bash
cd ax_convert
python run.py
```
输出示例如下：
![demo.jpg](../assets/minicpmo2_6/show_demo.jpg)

```bash
question1 = "What is the landform in the picture?"

answer1 = The landform in the picture is karst topography, characterized by steep limestone hills and cliffs. This type of landscape typically forms due to the dissolution of soluble rocks like limestone over time, creating unique shapes that are often seen in regions with significant geological activity.
------------------
question2 = "What should I pay attention to when traveling here?"
answer2 = When traveling to areas with karst topography, it's important to be mindful of the terrain. The landscape can present challenges such as uneven ground and potential rockfalls. Additionally, always respect local ecosystems by not disturbing flora or fauna. If you plan to explore further into these hills, consider bringing sturdy footwear for hiking.
"""
```

执行导出onnx脚本：
```bash
cd ax_convert
python export_onnx.py 
```
会在ax_convert目录上生成如下数据文件：
- siglip.onnx
- resampler.onnx
- embed_tokens.pth，这是llm里的词向量权重矩阵

并用onnxslim优化模型结构，使其更好的适配后续的puslar2编译部分。
```bash
onnxslim siglip.onnx siglip-slim.onnx

onnxslim resampler.onnx resampler-slim.onnx
```

可选：验证上述导出的onnx是否正确，执行脚本：
```bash
python run_siglip_onnx.py
```
如果输出如下，则说明模型导出成功：
```bash
The landform in the picture is karst topography, characterized by steep limestone hills and cliffs. This type of landscape typically forms due to the dissolution of soluble rocks like limestone over time, creating unique shapes that are often seen in regions with significant geological activity.
```

## 编译模型（ONNX->AXMODEL）
已提供必要的编译配置文件：
- `config_resampler.json`，其中字段`calibration_dataset`指向上一步中resampler.tar文件
- `config_siglip.json`, 其中字段`calibration_dataset`指向上一步中siglip.tar文件

生成编译模型需要的量化数据集，执行脚本：   
```bash
python prepare_calib.py
```
会在ax_convert目录两个量化数据集目录`calib_resampler_data`，`calib_siglip_data`，用于后续的模型量化，目录内容分别如下：
```bash
calib_resampler_data/
├── data_0.npy
├── data_10.npy
├── data_11.npy
├── data_12.npy
├── data_13.npy
├── data_14.npy
├── data_15.npy
├── data_16.npy
├── data_17.npy
├── data_18.npy
├── data_19.npy
├── data_1.npy
├── data_20.npy
├── data_21.npy
├── data_22.npy
├── data_23.npy
├── data_24.npy
├── data_25.npy
├── data_26.npy
├── data_27.npy
├── data_28.npy
├── data_29.npy
├── data_2.npy
├── data_30.npy
├── data_31.npy
├── data_3.npy
├── data_4.npy
├── data_5.npy
├── data_6.npy
├── data_7.npy
├── data_8.npy
├── data_9.npy
├── data.npy
└── resampler.tar  # 此tar文件用于resampler模型量化
=================
calib_siglip_data/
├── data_0.npy
├── data_10.npy
├── data_11.npy
├── data_12.npy
├── data_13.npy
├── data_14.npy
├── data_15.npy
├── data_16.npy
├── data_17.npy
├── data_18.npy
├── data_19.npy
├── data_1.npy
├── data_20.npy
├── data_21.npy
├── data_22.npy
├── data_23.npy
├── data_24.npy
├── data_25.npy
├── data_26.npy
├── data_27.npy
├── data_28.npy
├── data_29.npy
├── data_2.npy
├── data_30.npy
├── data_31.npy
├── data_3.npy
├── data_4.npy
├── data_5.npy
├── data_6.npy
├── data_7.npy
├── data_8.npy
├── data_9.npy
├── data.npy
└── siglip.tar  # 此tar文件用于siglip模型量化
```

[Pulsar2编译模型](https://npu.pages-git-ext.axera-tech.com/pulsar2-docs/user_guides_quick/quick_start_ax650.html)
```bash
# siglip 模型
pulsar2 build --input siglip-slim.onnx --config config_siglip.json --output_dir output_siglip --debug.dump_frontend_graph --output_name siglip.axmodel

# resampler 模型
pulsar2 build --input resampler-slim.onnx --config config_resampler.json --output_dir output_resampler --debug.dump_frontend_graph --output_name resampler.axmodel

# llm 模型
pulsar2 llm_build --input_path /data/wangjian/project/hf_cache/openbmb/MiniCPM-V-4  --output_path minicpm-v-4_axmodel/ --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 320 --parallel 32 --chip AX650
```
其中llm_build编译minicmp-v生成的目录内容如下：
```bash
minicpm-v-4_axmodel/
├── minicpmv_p320_l0_together.axmodel
├── minicpmv_p320_l10_together.axmodel
├── minicpmv_p320_l11_together.axmodel
├── minicpmv_p320_l12_together.axmodel
├── minicpmv_p320_l13_together.axmodel
├── minicpmv_p320_l14_together.axmodel
├── minicpmv_p320_l15_together.axmodel
├── minicpmv_p320_l16_together.axmodel
├── minicpmv_p320_l17_together.axmodel
├── minicpmv_p320_l18_together.axmodel
├── minicpmv_p320_l19_together.axmodel
├── minicpmv_p320_l1_together.axmodel
├── minicpmv_p320_l20_together.axmodel
├── minicpmv_p320_l21_together.axmodel
├── minicpmv_p320_l22_together.axmodel
├── minicpmv_p320_l23_together.axmodel
├── minicpmv_p320_l24_together.axmodel
├── minicpmv_p320_l25_together.axmodel
├── minicpmv_p320_l26_together.axmodel
├── minicpmv_p320_l27_together.axmodel
├── minicpmv_p320_l28_together.axmodel
├── minicpmv_p320_l29_together.axmodel
├── minicpmv_p320_l2_together.axmodel
├── minicpmv_p320_l30_together.axmodel
├── minicpmv_p320_l31_together.axmodel
├── minicpmv_p320_l3_together.axmodel
├── minicpmv_p320_l4_together.axmodel
├── minicpmv_p320_l5_together.axmodel
├── minicpmv_p320_l6_together.axmodel
├── minicpmv_p320_l7_together.axmodel
├── minicpmv_p320_l8_together.axmodel
├── minicpmv_p320_l9_together.axmodel
└── minicpmv_post.axmodel
```

## AX650端板推理
将必要的运行文件scp到650的板子上：
- run_axmodel.py，板端执行文件
- embed_tokens.pth，minicpm-v中用到的词嵌入矩阵
- output_siglip/siglip.axmodel
- output_resampler/resampler.axmodel 
- minicpm-v-4_axmodel，minicpm-v编译的目录
- MiniCPM-V-4 repo里面的*.py,*.json文件

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

执行脚本：
```bash
python3 run_axmodel.py -i show_demo.jpg -q "What is the landform in the picture?"
```
输入图片：
输出示例如下：
![demo.jpg](../assets/minicpmo2_6/show_demo.jpg)

```bash
question1 = "What is the landform in the picture?"

answer1 = The landform in the picture is a karst topography, characterized by its unique and dramatic appearance with steep limestone cliffs rising from the water' s surface. This type of landscape is commonly found in regions with significant geological activity, such as China's Li River.
```