import torch
import argparse
import onnx
import onnxruntime as ort
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor


class SigLIPWarpper(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.vpm = model.vpm
        # self.resampler = model.resampler
    
    def get_position_ids(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor, tgt_sizes: torch.IntTensor=None):
        batch_size = pixel_values.size(0)
        
        max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
        max_nb_patches_h, max_nb_patches_w = max_im_h // self.vpm.embeddings.patch_size, max_im_w // self.vpm.embeddings.patch_size
        boundaries = torch.arange(1 / self.vpm.embeddings.num_patches_per_side, 1.0, 1 / self.vpm.embeddings.num_patches_per_side)
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

            pos_ids = (bucket_coords_h[:, None] * self.vpm.embeddings.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids
            # position_ids[batch_idx] = torch.where(p_attn_mask.view(-1).cpu(), pos_ids, position_ids[batch_idx])
    
        return position_ids
    
    def vision_embed_forward(self, pixel_values: torch.FloatTensor, position_ids: torch.IntTensor) -> torch.Tensor:

        patch_embeds = self.vpm.embeddings.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        embeddings = embeddings + self.vpm.embeddings.position_embedding(position_ids)
        return embeddings
    
    def forward(self, all_pixel_values, position_ids):
        hidden_states = self.vision_embed_forward(all_pixel_values, position_ids)
        
        encoder_outputs = self.vpm.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )
        
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.vpm.post_layernorm(last_hidden_state)
        
        vision_embedding = last_hidden_state
        return vision_embedding
        
    
class ResamplerWarpper(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.resampler = model.resampler
    
    def pre_forward(self, x, tgt_sizes=None):
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]
        
        self.resampler._adjust_pos_cache(tgt_sizes, device=device)

        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self.resampler.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype))  # patches * D
            key_padding_mask[i, patch_len[i]:] = True

        pos_embed = torch.nn.utils.rnn.pad_sequence(
            pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)  # BLD => L * B * D
        
        setattr(self, 'pos_embed', pos_embed)
        setattr(self, 'key_padding_mask', key_padding_mask)
        
    def forward(self, x, tgt_sizes=None):
        pos_embed = getattr(self, "pos_embed")
        key_padding_mask = getattr(self, "key_padding_mask")
        
        bs = x.shape[0]
        x = self.resampler.kv_proj(x)  # B * L * D
        x = self.resampler.ln_kv(x).permute(1, 0, 2)  # L * B * D

        q = self.resampler.ln_q(self.resampler.query)  # Q * D

        out = self.resampler.attn(
            self.resampler._repeat(q, bs),  # Q * B * D
            x + pos_embed,  # L * B * D +  L * B * D
            x,
            key_padding_mask=key_padding_mask)[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.resampler.ln_post(x)
        x = x @ self.resampler.proj
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument("-m", "--model", default="/data/wangjian/project/hf_cache/openbmb/MiniCPM-V-4", type=str, help="hugging fance model path")
    parser.add_argument("--imgsize", type=int, default=448, help="onnx input image size")
    parser.add_argument("--siglip_onnx_save_dir", type=str, default='siglip.onnx', help="siglip onnx model save path")
    parser.add_argument("--resampler_onnx_save_dir", type=str, default='resampler.onnx', help="resampler onnx model save path")
    parser.add_argument("--embed_token_save_dir", type=str, default='embed_tokens.pth', help="resampler onnx model save path")

    args = parser.parse_args()

    torch.manual_seed(100)

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, # or openbmb/MiniCPM-o-2_6
        attn_implementation='sdpa', torch_dtype=torch.float32) # sdpa or flash_attention_2, no eager
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True) # or openbmb/MiniCPM-o-2_6

    vpm = SigLIPWarpper(model)
    device = model.device
    all_pixel_values = torch.rand((1, 3, 14, 14336), dtype=torch.float32).to(device)  # 14336 = 14 * (448/14) * (448/14)
    patch_attention_mask = torch.ones(1, 1, 1024).bool().to(device)
    tgt_sizes = torch.tensor([[32, 32]],).to(device)
    position_ids = vpm.get_position_ids(all_pixel_values, patch_attention_mask, tgt_sizes)
    torch.onnx.export(
        vpm,
        (all_pixel_values, position_ids),
        args.siglip_onnx_save_dir,
        opset_version=17, # 14
        do_constant_folding=True,
        input_names=["all_pixel_values", "position_ids"],
        output_names=["vision_embedding"],
        verbose=False,
    )
    
    resampler = ResamplerWarpper(model)
    vision_embedding = torch.rand((1, 1024, 1152), dtype=torch.float32).to(device)
    resampler.pre_forward(vision_embedding, tgt_sizes)
    torch.onnx.export(
        resampler,
        (vision_embedding, tgt_sizes),
        args.resampler_onnx_save_dir,
        opset_version=17, # 14
        do_constant_folding=False,
        input_names=["vision_embedding", "tgt_sizes"],
        output_names=["vision_embedding_out"],
        verbose=False,
    )
        
    torch.save(model.llm.model.embed_tokens, args.embed_token_save_dir)