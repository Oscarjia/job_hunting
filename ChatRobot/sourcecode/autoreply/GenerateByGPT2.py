# ========================================
# Author: Jiang Xiaotian
# Email: jxt441621944@163.com
# Copyright: lorewalkeralex @ 2020
# ========================================

import torch
from transformers.modeling_gpt2 import GPT2LMHeadModel
from transformers import BertTokenizer


# 根据GPT2生成回复
class GPT2Generate:
    def __init__(self, model_path, vocab_path):
        torch.cuda.empty_cache()
        use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.tokenizer = BertTokenizer(vocab_file=vocab_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, inputQ):
        # sentence embedding
        input_str_ids = self.tokenizer.encode(inputQ)
        input_ids = [self.tokenizer.cls_token_id] + input_str_ids + [self.tokenizer.sep_token_id]
        input_tensor = torch.tensor(input_ids).long().to(self.device)
        # generate
        response = ''
        while True:
            with torch.no_grad():
                outputs = self.model(input_ids=input_tensor)
            next_token_logits = outputs[0][-1, :]
            predicted_index = torch.argmax(next_token_logits).item()
            if predicted_index == self.tokenizer.sep_token_id:
                break
            predicted_word = self.tokenizer.decode([predicted_index])
            response += predicted_word
            input_tensor = torch.cat((input_tensor, torch.tensor([predicted_index]).to(self.device)), dim=0)
        return response