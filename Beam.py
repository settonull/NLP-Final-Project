# Adapted from fairseq implementation https://github.com/pytorch/fairseq/blob/master/fairseq/search.py

import torch

class Beam(object):

    def __init__(self, lang, device):
        self.pad = lang.PAD_IDX
        self.sos = lang.SOS_IDX
        self.eos = lang.EOS_IDX
        self.unk = lang.UNK_IDX

        self.scores_buf = None
        self.indices_buf = None
        self.beams_buf = None


    def step(self, step, logprobs, scores):
        batch_size, beam_size, vocab_size = logprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            logprobs = logprobs[:, ::beam_size, :].contiguous()
            
            ##Init buffers
            if self.scores_buf is None:
                self.scores_buf = t.new()
                self.indices_buf = torch.LongTensor().to(device= logprobs.device)
                self.beams_buf = torch.LongTensor().to(device= logprobs.device)

        else:
            # make probs contain cumulative scores for each hypothesis
            logprobs.add_(scores[:, :, step - 1].unsqueeze(-1))

        torch.topk(
            logprobs.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                logprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
            out=(self.scores_buf, self.indices_buf),
        )
        torch.div(self.indices_buf, vocab_size, out=self.beams_buf)
        self.indices_buf.fmod_(vocab_size)
        return self.scores_buf, self.indices_buf, self.beams_buf

    
    
    
    
    