# Inspired by the basic structure of the fairseq implementation https://github.com/pytorch/fairseq/blob/master/fairseq/search.py
# Drives the actual search part of beam search, the decoding and other pieces take place in the beam_search function in
# translator.py

import torch

class Beam(object):

    def __init__(self, lang, device):
        self.pad = lang.PAD_IDX
        self.sos = lang.SOS_IDX
        self.eos = lang.EOS_IDX
        self.unk = lang.UNK_IDX
        self.vocab_size = lang.n_words

        self.hist_scores_buf = None
        self.indices = torch.LongTensor().to(device= self.device)
        self.beams = torch.LongTensor().to(device= self.device)


    def step(self, step, logprobs, scores):
        ##Init buffers
        if self.scores_buf is None:
            self.scores_buf = logprobs.new()
        
        batch_size, beam_size, vocab_size = logprobs.size()
        cand_size = 2* beam_size

        if step == 0:
            logprobs = logprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            logprobs.add_(scores[:, :, step - 1].unsqueeze(-1))

        torch.topk(logprobs.view(batch_size, -1),
            k=min(cand_size, logprobs.view(batch_size, -1).size(1) - 1),
            out=(self.scores_buf, self.indices_buf))
        
        torch.div(self.indices_buf, vocab_size, out=self.beams_buf)
        self.indices_buf.fmod_(vocab_size)
        return self.scores_buf, self.indices_buf, self.beams_buf\
    
    
    
    
    