from engine import Engine, AnswerLog
import json
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import datetime
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from typing import List
from documents.document import Library, Text, Chunk, Prompt
from accelerate import infer_auto_device_map, init_empty_weights

class MistralEngine(Engine):
    def __init__(self, lora_path, parameters, system_description, cpu_offload=True):
        Engine.__init__(self, "mistral", parameters)
        self.system_description = system_description
        self.prices_prompt = {
            "mistral": 0
        }
        self.prices_completion = {
            "mistral": 0
        }

        model_name_or_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        config.max_position_embeddings = 1024

        quantization_config = BitsAndBytesConfig(
            llm_int8_enable_fp32_cpu_offload=cpu_offload,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            load_in_4bit=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        with init_empty_weights():
            self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            )
        device_map = infer_auto_device_map(self.hf_model)
        device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1.self_attn': 0, 'model.layers.1.block_sparse_moe.gate': 0, 'model.layers.1.block_sparse_moe.experts.0': 0, 'model.layers.1.block_sparse_moe.experts.1': 0, 'model.layers.1.block_sparse_moe.experts.2': 0, 'model.layers.1.block_sparse_moe.experts.3': 0, 'model.layers.1.block_sparse_moe.experts.4': 0, 'model.layers.1.block_sparse_moe.experts.5': 0, 'model.layers.1.block_sparse_moe.experts.6': 0, 'model.layers.1.block_sparse_moe.experts.7.w1': 0, 'model.layers.1.block_sparse_moe.experts.7.w2': 0, 'model.layers.1.block_sparse_moe.experts.7.w3': 'cpu', 'model.layers.1.block_sparse_moe.experts.7.act_fn': 'cpu', 'model.layers.1.input_layernorm': 'cpu', 'model.layers.1.post_attention_layernorm': 'cpu', 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 'cpu', 'model.layers.8': 'cpu', 'model.layers.9': 'cpu', 'model.layers.10': 'cpu', 'model.layers.11': 'cpu', 'model.layers.12': 'cpu', 'model.layers.13': 'cpu', 'model.layers.14': 'cpu', 'model.layers.15': 'cpu', 'model.layers.16': 'cpu', 'model.layers.17': 'cpu', 'model.layers.18': 'cpu', 'model.layers.19': 'cpu', 'model.layers.20': 'cpu', 'model.layers.21': 'cpu', 'model.layers.22': 'cpu', 'model.layers.23': 'cpu', 'model.layers.24': 'cpu', 'model.layers.25': 'cpu', 'model.layers.26': 'cpu', 'model.layers.27.self_attn': 'cpu', 'model.layers.27.block_sparse_moe.gate': 'cpu', 'model.layers.27.block_sparse_moe.experts.0': 'cpu', 'model.layers.27.block_sparse_moe.experts.1': 'cpu', 'model.layers.27.block_sparse_moe.experts.2.w1': 'cpu', 'model.layers.27.block_sparse_moe.experts.2.w2': 'disk', 'model.layers.27.block_sparse_moe.experts.2.w3': 'disk', 'model.layers.27.block_sparse_moe.experts.2.act_fn': 'disk', 'model.layers.27.block_sparse_moe.experts.3': 'disk', 'model.layers.27.block_sparse_moe.experts.4': 'disk', 'model.layers.27.block_sparse_moe.experts.5': 'disk', 'model.layers.27.block_sparse_moe.experts.6': 'disk', 'model.layers.27.block_sparse_moe.experts.7': 'disk', 'model.layers.27.input_layernorm': 'disk', 'model.layers.27.post_attention_layernorm': 'disk', 'model.layers.28': 'disk', 'model.layers.29': 'disk', 'model.layers.30': 'disk', 'model.layers.31': 'disk', 'model.norm': 'disk', 'lm_head': 'disk'}
        print("@@@@@@@@@@@@@")
        print(device_map)
        print("-------------")
        self.hf_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map=device_map,
        offload_folder="./offload"
        )
        # peft_model_id = lora_path
        # self.hf_model.load_adapter(peft_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def query(self, prompt: str, max_tokens: int = 256, temperature: float = 0):
        self.n_queries += 1
        prompt = f"<s>### Instruction: {self.system_description}\n### Input:\n{prompt}\n### Response:"
        prompt = Prompt(prompt)
        encoded_input = self.tokenizer(prompt.content,  return_tensors="pt", add_special_tokens=True)
        model_inputs = encoded_input.to('cuda')
        generated_ids = self.hf_model.generate(**model_inputs, max_new_tokens=max_tokens, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
        decoded_output = self.tokenizer.batch_decode(generated_ids)
        answer = decoded_output[0].split("### Response:")[1].replace("</s>", "")
        if answer[0] == "\n":
            answer = answer[1:]
        answer = Text(answer)
        answer_log = AnswerLog(
            prompt, answer, self.model, self.prices_prompt, self.prices_completion
        )
        if self.parameters["logs"]["autosave"]:
            with open(
                f"{self.parameters['logs']['save_path']}/{self.run_id}/{self.n_queries}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(answer_log.data, f, ensure_ascii=False, indent=4)
        self.logs.append(answer_log)
        return answer