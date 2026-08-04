# from mistralai import Mistral
from mistralai.client import Mistral
import openai
import os
from env import OPENAI_API_KEY
import torch

class llm_client:
    def __init__(self, LLM='llama', temperature=0, max_tries=1, model=None, tokenizer=None):
        
        self.LLM = LLM
        self.temperature = temperature
        self.max_tries = max_tries
        self.model = model
        self.tokenizer = tokenizer

        
        if 'llama' in self.LLM:
            if self.model is None or self.tokenizer is None:
                raise ValueError("For llama, pass model and tokenizer from get_pretrained_llm().")
        # if 'gpt' in self.LLM:     
        #     openai.api_key = OPENAI_API_KEY
    
        # elif 'mistral' in self.LLM or 'mixtral' in self.LLM:
        #     api_key = os.environ.get("MISTRALAIKEY")
        #     self.client = Mistral(api_key=api_key)
        else:                     
            raise ValueError(f"model ({self.LLM}) still not implemented")
                                  
    def invoke(self, question, temperature=0):
        # if 'gpt' in self.LLM:
        #     response = openai.chat.completions.create(
        #         model=self.LLM,
        #         messages=[
        #             {"role": "system", "content": "You are a helpful AI assistant."},
        #             {"role": "user", "content": question},
        #         ],
        #         temperature=temperature
        #     )
        #     answer = response.choices[0].message.content

        if 'llama' in self.LLM:
            device = next(self.model.parameters()).device

            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": question},
            ]

            prompt = ""

            for msg in messages:
                prompt += f"{msg['role']}: {msg['content']}\n"

            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else None,
                    max_new_tokens=512,
                    repetition_penalty=1.1,
                )

            generated = outputs[0][input_len:]
            answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()


        # elif 'mistral' in self.LLM or 'mixtral' in self.LLM:
        #     chat_response = self.client.chat.complete(
        #         model = self.LLM,
        #         messages = [
        #             {"role": "system", "content": "You are a helpful AI assistant."},
        #             {"role": "user", "content": question},
        #         ],
        #         temperature=temperature
        #     )
        #     answer = chat_response.choices[0].message.content
        else:
            raise ValueError("model still not implemented")
        return answer
    
    def parse_answer(self, answer):
        start_tag = "<answer>"
        end_tag = "</answer>"
        start_index = answer.find(start_tag) + len(start_tag)
        end_index = answer.find(end_tag)
        if start_index != -1 and end_index != -1:
            answer = answer[start_index:end_index].strip()
        return answer

    # This function is intended to be used only for asking the model to provide the causal relationship between two concepts
    def invoke_with_retry(self, question):
        tries = 0
        answers = {}
        while tries < self.max_tries:
            answer = self.invoke(question, self.temperature)
            answer = self.parse_answer(answer)
            if answer in answers.keys():
                answers[answer] += 1
            else:
                answers[answer] = 1
            tries += 1
        return max(answers, key=answers.get)