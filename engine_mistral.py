from engine import Engine, AnswerLog
import json
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import datetime

from typing import List
from documents.document import Library, Text, Chunk, Prompt

class MistralEngine(Engine):
    def __init__(self, parameters, system_description):
        Engine.__init__(self, "mistral", parameters)
        self.system_description = system_description
        self.prices_prompt = {
            "mistral": 0
        }
        self.prices_completion = {
            "mistral": 0
        }

        # model = get_peft_model("./mistral_function_calling_v0", peft_config)
        model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        peft_model_id = "./mistral_function_calling_v0"

        self.hf_model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
        self.hf_model.load_adapter(peft_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def query(self, prompt: str, max_tokens: int = 256, temperature: float = 0):
        prompt = f"<s>### Instruction: {self.system_description}\n### Input:\n{prompt}\n### Response:"
        encoded_input = self.tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
        model_inputs = encoded_input.to('cuda')

        generated_ids = self.hf_model.generate(**model_inputs, max_new_tokens=max_tokens, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)

        decoded_output = self.tokenizer.batch_decode(generated_ids)
        answer = decoded_output[0].replace(prompt, "")
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