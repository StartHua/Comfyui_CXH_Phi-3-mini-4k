import os
import torch
import folder_paths
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class Phi_minni:
    def __init__(self):
        self.model_checkpoint = None
        self.model_cache = None
        self.tokenizer = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "instruction": ("STRING",{"default": '', "multiline": True}),
                "prompt": ("STRING",{"default": '', "multiline": True}),
                "model": (["Phi-3-mini-4k-instruct", "Phi-3-mini-128k-instruct"],),
                "temperature": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 500, "min": 100, "max": 2000, "step": 500}),
                
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "CXH/GPT"

    def inference(self,instruction,prompt,model, temperature,max_new_tokens):
        # 下载本地
        model_id = f"microsoft/{model}"
        model_checkpoint = os.path.join(folder_paths.models_dir, 'microsoft', os.path.basename(model_id))
        if not os.path.exists(model_checkpoint):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)
            
        torch.random.manual_seed(0)
        
        # 加载模型
        if self.model_checkpoint != model_checkpoint:
            self.model_checkpoint = model_checkpoint   
            self.model_cache = AutoModelForCausalLM.from_pretrained(
            model_checkpoint, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True, 
        )
            self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        print("========")
        print(instruction)
        print(prompt)
        # text 
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ]

        pipe = pipeline(
            "text-generation",
            model=self.model_cache,
            tokenizer=self.tokenizer,
        )

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
            "temperature":temperature,
            "do_sample": False,
        }
       
        
        output = pipe(messages, **generation_args)
        print("======")
        print(output)
        result = output[0]['generated_text']
        print(result)
        return (result,)
        

NODE_CLASS_MAPPINGS = {
    "Phi_minni": Phi_minni,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Phi_minni": "Phi_minni_CXH",
}