from ldm_patched.modules import sd1_clip
import ldm_patched.modules.text_encoders.t5
import ldm_patched.modules.model_management
from transformers import T5TokenizerFast
import torch
import os

class T5XXLModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None):
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_config_xxl.json")
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"end": 1, "pad": 0}, model_class=ldm_patched.modules.text_encoders.t5.T5)
        self.device = device

    def encode_token_weights(self, token_weight_pairs):
        if not token_weight_pairs:
            # Return empty tensors if no tokens
            return torch.empty(1, 0, self.transformer.config.hidden_size, device=self.device), None

        tokens = [t[0] for t in token_weight_pairs]
        weights = [t[1] for t in token_weight_pairs]
        tokens = torch.tensor([tokens], device=self.device)
        weights = torch.tensor([weights], device=self.device)
        return self(tokens)

class T5XXLTokenizer:
    def __init__(self, embedding_directory=None):
        try:
            self.tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-xxl", legacy=False)
        except Exception as e:
            print(f"Error loading T5XXL tokenizer: {e}")
            print("Falling back to T5TokenizerFast without pretrained weights")
            self.tokenizer = T5TokenizerFast()
        self.embedding_directory = embedding_directory

    def tokenize_with_weights(self, text: str, return_word_ids=False):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return [(t, 1.0) for t in tokens]

class FluxTokenizer:
    def __init__(self, embedding_directory=None):
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory)
        self.t5xxl = T5XXLTokenizer(embedding_directory=embedding_directory)

    def tokenize_with_weights(self, text:str, return_word_ids=False):
        out = {}
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids)
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text, return_word_ids)
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_l.untokenize(token_weight_pair)

    def state_dict(self):
        return {}


class FluxClipModel(torch.nn.Module):
    def __init__(self, dtype_t5=None, device="cpu", dtype=None):
        super().__init__()
        dtype_t5 = ldm_patched.modules.model_management.pick_weight_dtype(dtype_t5, dtype, device)
        self.clip_l = sd1_clip.SDClipModel(device=device, dtype=dtype)
        self.t5xxl = T5XXLModel(device=device, dtype=dtype_t5)
        self.dtypes = set([dtype, dtype_t5])
        try:
            self.tokenizer = FluxTokenizer()
        except Exception as e:
            print(f"Error initializing FluxTokenizer: {e}")
            print("Falling back to CLIP-L tokenizer only")
            self.tokenizer = sd1_clip.SDTokenizer()
        self.device = device

    def encode_embedding_init_text(self, init_text, nvpt):
        tokens = self.tokenizer.clip_l.tokenizer(init_text)
        if isinstance(tokens, dict):
            tokens = tokens['input_ids']
        elif not isinstance(tokens, list) or len(tokens) == 0:
            print("Warning: No tokens generated!")
            return None  # Early return, or handle the case as needed

        tokens = tokens[:nvpt]
        while len(tokens) < nvpt:
            tokens.append(self.tokenizer.clip_l.id_end)
        tokens = torch.asarray([tokens])
        return self.clip_l.encode_with_transformer(tokens)

    def forward(self, text):
        if isinstance(text, str):
            text = [text]
        
        # CLIP Processing
        clip_tokens = self.tokenizer.clip_l.tokenize_with_weights(text[0])  # Tokenize the first text
        clip_tokens = [[t[0] for t in clip_tokens[0]]]  # Prepare for CLIP input
        if not clip_tokens or not clip_tokens[0]:  # Check for empty tokens
            print("Warning: No CLIP tokens generated.")
            return None, None  # or raise an exception

        clip_output, clip_pooled = self.clip_l(clip_tokens)
        
        # T5 Processing
        t5_tokens = self.tokenizer.t5xxl.tokenize_with_weights(text[0])
        if not t5_tokens:  # Check if T5 tokenizer returns valid tokens
            print("Warning: No T5 tokens generated.")
            return None, None  # or raise an exception

        t5_output, t5_pooled = self.t5xxl.encode_token_weights(t5_tokens)

        # Ensure both outputs are non-empty before concatenating
        if clip_output.size(0) == 0:
            print("Warning: CLIP output is empty.")
            return None, None  # or raise an exception

        if t5_output.size(0) == 0:
            print("Warning: T5 output is empty.")
            return None, None  # or raise an exception

        # Combine CLIP and T5 outputs
        combined_output = torch.cat([clip_output, t5_output], dim=-1)
        combined_pooled = torch.cat([clip_pooled, t5_pooled], dim=-1) if clip_pooled is not None and t5_pooled is not None else None
        
        return combined_output, combined_pooled

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.t5xxl.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_l.reset_clip_options()
        self.t5xxl.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        clip_tokens = self.tokenizer.clip_l.tokenize_with_weights(token_weight_pairs)
        clip_output = self.clip_l.encode_token_weights(clip_tokens)
        
        t5_tokens = self.tokenizer.t5xxl.tokenize_with_weights(token_weight_pairs)
        t5_output = self.t5xxl.encode_token_weights(t5_tokens)
        
        combined_output = torch.cat([clip_output, t5_output], dim=-1)
        return combined_output

    def load_sd(self, sd):
        if "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        else:
            return self.t5xxl.load_sd(sd)

def flux_clip(dtype_t5=None):
    class FluxClipModel_(FluxClipModel):
        def __init__(self, device="cpu", dtype=None):
            super().__init__(dtype_t5=dtype_t5, device=device, dtype=dtype)
    return FluxClipModel_