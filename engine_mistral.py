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

        model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        config.max_position_embeddings = 8096
        quantization_config = BitsAndBytesConfig(
            llm_int8_enable_fp32_cpu_offload=cpu_offload,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            load_in_4bit=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
        offload_folder="./offload"
        )

        peft_model_id = lora_path
        self.hf_model.load_adapter(peft_model_id)
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