import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Sampler:
    '''
    LLM token sampler with KV caching
    '''
    def __init__(self, model_name, topk=40):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.cache = None
        self.topk = topk

    @torch.no_grad()
    def start(self, context: str):
        input_ids = self.tokenizer(context, return_tensors="pt").input_ids
        out = self.model(input_ids, use_cache=True)
        logits = out.logits[0, -1]  # last position's logits
        self.cache = out.past_key_values
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_ids = torch.topk(probs, self.topk)
        return topk_probs.numpy(), topk_ids.numpy()

    @torch.no_grad()
    def step(self, token_id: int):
        '''
        Reuse KV cache to append single new token to context
        '''
        id = torch.tensor([[token_id]]) # (1, 1)
        out = self.model(id, use_cache=True, past_key_values=self.cache)
        logits = out.logits[0, -1]  # last position's logits
        self.cache = out.past_key_values
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_ids = torch.topk(probs, self.topk)
        return topk_probs.numpy(), topk_ids.numpy()