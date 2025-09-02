import os
import torch
import argparse
import tarfile
from copy import deepcopy
import onnxruntime as ort
from PIL import Image
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoConfig


class MiniCPMV(torch.nn.Module):
    def __init__(self, siglip_onnx_path, resampler_onnx_path, embed_token_path, config) -> None:
        super().__init__()
        self.vpm = ort.InferenceSession(siglip_onnx_path)
        self.resampler = ort.InferenceSession(resampler_onnx_path)
        self.embed_tokens = torch.load(embed_token_path, weights_only=False)
        self.config = config
    
    def get_position_ids(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor, tgt_sizes: torch.IntTensor=None):
        batch_size = pixel_values.size(0)
        
        max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
        max_nb_patches_h, max_nb_patches_w = max_im_h // self.config.vision_config.patch_size, max_im_w // self.config.vision_config.patch_size
        num_patches_per_side = self.config.vision_config.image_size // self.config.vision_config.patch_size
        boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
        position_ids = torch.full(
            size=(
                batch_size,
                max_nb_patches_h * max_nb_patches_w,
            ),
            fill_value=0,
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            if tgt_sizes is not None:
                nb_patches_h = tgt_sizes[batch_idx][0]
                nb_patches_w = tgt_sizes[batch_idx][1]
            else:
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids
            # position_ids[batch_idx] = torch.where(p_attn_mask.view(-1).cpu(), pos_ids, position_ids[batch_idx])
    
        return position_ids
    
    @torch.no_grad()
    def get_vllm_embedding(self, data):
        if 'vision_hidden_states' not in data:
            dtype = torch.float32
            device = "cpu"
            tgt_sizes = data['tgt_sizes']
            pixel_values_list = data['pixel_values']
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,
                                                                   padding_value=0.0)
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                for i in range(B):
                    patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = self.config.vision_batch_size
                all_pixel_values = all_pixel_values.type(dtype).to(device=device)
                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        tmp_hs = self.vpm(all_pixel_values[start_idx:end_idx], patch_attention_mask=patch_attn_mask[start_idx:end_idx], tgt_sizes=tgt_sizes[start_idx:end_idx]).last_hidden_state
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else: # 走这里
                    position_ids = self.get_position_ids(all_pixel_values, patch_attention_mask=patch_attn_mask, tgt_sizes=tgt_sizes)
                    siglip_inputs = {
                        "all_pixel_values": all_pixel_values.numpy(),
                        "position_ids": position_ids.numpy(),
                    }
                    vision_embedding = self.vpm.run(None, input_feed=siglip_inputs)[0]
                resampler_inputs = {
                    "vision_embedding": vision_embedding,
                    # "tgt_sizes": tgt_sizes.type(torch.int64).numpy(),
                }
                vision_embedding = self.resampler.run(None, input_feed=resampler_inputs)[0]
                vision_embedding = torch.from_numpy(vision_embedding)
                
                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start: start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else: # no image
                if self.training:
                    dummy_image = torch.zeros(
                        (1, 3, 224, 224),
                        device=device, dtype=dtype
                    )
                    tgt_sizes = torch.Tensor([[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)

        else:
            vision_hidden_states = data['vision_hidden_states']       
        
        vllm_embedding = self.embed_tokens(data['input_ids'])
        
        vision_hidden_states = [i.type(vllm_embedding.dtype) if isinstance(
            i, torch.Tensor) else i for i in vision_hidden_states]

        bs = len(data['input_ids'])
        device = vllm_embedding.device
        embed_dim = vllm_embedding.shape[-1]

        new_vllm_embeddings = []
        
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            cur_vllm_emb = vllm_embedding[i]

            if len(cur_vs_hs) == 0:
                new_vllm_embeddings.append(cur_vllm_emb)
                continue
                
            cur_image_bound = data['image_bound'][i]

            if len(cur_image_bound) > 0:
                image_indices = torch.stack([
                    torch.arange(r[0], r[1], dtype=torch.long) 
                    for r in cur_image_bound
                ], dim=0).flatten().to(device)

                indices_expanded = image_indices.view(-1, 1).expand(-1, embed_dim)
                vision_features = cur_vs_hs.view(-1, embed_dim)
                
                updated_emb = cur_vllm_emb.scatter(0, indices_expanded, vision_features)
                new_vllm_embeddings.append(updated_emb)
            elif self.training:
                dummy_term = cur_vs_hs[0].sum() * 0 
                new_vllm_embeddings.append(cur_vllm_emb + dummy_term)
            else:
                new_vllm_embeddings.append(cur_vllm_emb)

        vllm_embedding = torch.stack(new_vllm_embeddings, dim=0)
        np.save("vision_embedding_onnx.npy", vllm_embedding.detach().cpu().numpy())
        np.save("vision_hidden_states_onnx.npy", [v.cpu().numpy() for v in vision_hidden_states])
        return vllm_embedding, vision_hidden_states

    def _decode(self, inputs_embeds, tokenizer, attention_mask, decode_text=False, **kwargs):
        model = AutoModel.from_pretrained(hf_model_path, trust_remote_code=True, # or openbmb/MiniCPM-o-2_6
            attn_implementation='sdpa', torch_dtype=torch.float32) # sdpa or flash_attention_2, no eager
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in model.terminators]
        output = model.llm.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=0,
            eos_token_id=terminators,
            attention_mask=attention_mask,
            **kwargs
        )
        if decode_text:
            return model._decode_text(output, tokenizer)
        return output
    

if  __name__ == '__main__':
    hf_model_path = "/data/wangjian/project/hf_cache/openbmb/MiniCPM-V-4"
    img_path = "/data/wangjian/project/MiniCPM-o/assets/minicpmo2_6/show_demo.jpg"
    image = Image.open(img_path).convert('RGB').resize((448, 448)) 
    question = "What is the landform in the picture?"

    msgs = [{'role': 'user', 'content': [image, question]}]
    
    resampler_onnx = "/data/wangjian/project/MiniCPM-o/ax_convert/resampler-slim.onnx"
    siglip_onnx = "/data/wangjian/project/MiniCPM-o/ax_convert/siglip-slim.onnx"
    embed_token_path = "/data/wangjian/project/MiniCPM-o/ax_convert/embed_tokens.pth"
    
    processor = AutoProcessor.from_pretrained(hf_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
    
    processor.image_processor.slice_mode = False # 不对图像做切分操作
    
    model = MiniCPMV(siglip_onnx, resampler_onnx, embed_token_path, config)
    msgs_list = [msgs]

    prompts_lists = []
    input_images_lists = []
    for msgs in msgs_list:
        copy_msgs = deepcopy(msgs)
        images = []
        for i, msg in enumerate(copy_msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["user", "assistant"]
            if i == 0:
                assert role == "user", "The role of first msg should be user"
            if isinstance(content, str):
                content = [content]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_msgs.append(c)
            msg["content"] = "\n".join(cur_msgs)

        prompts_lists.append(processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True))
        input_images_lists.append(images)
    
    inputs = processor(
            prompts_lists, 
            input_images_lists, 
            max_slice_nums=None,
            use_image_id=None,
            return_tensors="pt", 
            max_length=32768
    )
    generation_config = {
        "top_p": 0.8,
        "top_k": 100,
        "temperature": 0.7,
        "do_sample": True,
        "repetition_penalty": 1.05
    }
    inputs.pop("image_sizes")
    
    model_inputs = {
        "input_ids": inputs.input_ids,
        "image_bound": inputs.image_bound,
    }
    model_inputs["pixel_values"] = inputs.pixel_values
    model_inputs['tgt_sizes'] = inputs.tgt_sizes
    
    model_inputs["inputs_embeds"], vision_hidden_states = model.get_vllm_embedding(model_inputs)

    result = model._decode(model_inputs["inputs_embeds"], tokenizer, inputs.attention_mask, decode_text=True, max_new_tokens=2048, **generation_config)
    print(result)