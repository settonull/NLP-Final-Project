from io import open
import random
import time
import math
import nltk
import Beam

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import copy
from collections import Counter
import tqdm

'''
DS-GA 1011 Final Project
Seq2Seq With Attention Translatoion Model

Much code adapted from Lab8 and Pytorch.org examples.
'''
from torch.utils.data import Dataset

# doesn't actually get called I don't think ?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# a bit of a hack, this must be called first so all methods have access to our device
def init_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Language:

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = ["<PAD>", "<SOS>", "<EOS>","<UNK>"]
        self.n_words = len(self.index2word)  # Count SOS and EOS, etc


    def build_vocab(self, all_tokens, max_vocab_size):
        token_counter = Counter(all_tokens)
        vocab, count = zip(*token_counter.most_common(max_vocab_size - self.n_words))
        id2token = list(vocab)
        self.word2index = dict(zip(vocab, range(self.n_words, self.n_words + len(vocab))))
        self.index2word.extend(id2token)
        self.n_words = len(self.index2word)



def indexSentences(lang, sentences):
    indexes = []
    lang_words = frozenset(lang.word2index.keys())
    for sentence in sentences:
        index = [lang.word2index[word] if word in lang_words else Language.UNK_IDX for word in sentence]
        index.append(Language.EOS_IDX)
        indexes.append(index)
    return indexes

def loadFile(lang_file):
    print("Loading data from ", lang_file, '... ', end='')

    sentences = []
    all_words = []

    with open(lang_file, 'r', encoding='utf-8') as f:

        for sentence in f:
            sent = []
            for word in sentence.strip().split(' '):
                sent.append(word)
                all_words.append(word)
            sentences.append(sent)
    print(len(sentences), 'sentences with', len(all_words), 'words', flush=True)
    return sentences, all_words

def loadData(lang_dir : str, lang_type, max_vocab):

    lang_file = lang_dir + 'train.tok.' + lang_type
    train_sentences, all_words = loadFile(lang_file)

    #right now we are just using our train set to define the vocab
    lang_map = Language(lang_type)
    lang_map.build_vocab(all_words, max_vocab)

    print("Created vocab for ", lang_type, 'of', lang_map.n_words, 'words')

    lang_file = lang_dir + 'dev.tok.' + lang_type
    val_sentences, all_words = loadFile(lang_file)

    lang_file = lang_dir + 'test.tok.' + lang_type
    test_sentences, all_words = loadFile(lang_file)

    return lang_map, train_sentences, val_sentences, test_sentences

def cleanSentences(input_sentences, output_sentences):

    removeList = []
    for i in range(max(len(input_sentences), len(output_sentences))):
        if (len(input_sentences[i]) == 1) and input_sentences[i][0] == '':
            removeList.append(i)
        if (len(output_sentences[i]) == 1) and output_sentences[i][0] == '' and i not in removeList:
            removeList.append(i)

    removeList.reverse()
    for i in removeList:
        del input_sentences[i]
        del output_sentences[i]

    if len(removeList) > 0:
        print("Removed", len(removeList), "blank sentences, lists now", len(input_sentences), len(output_sentences))
    #print("Sentence Lists:", len(input_sentences), len(output_sentences))

class PairsDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, lang1_list, lang2_list , MAX_SENTENCE_LENGTH):
        """
        @param data_list: list of character
        @param target_list: list of targets

        """
        self.lang1_idx = []
        self.lang1_len = []
        self.lang2_idx = []
        self.lang2_len = []

        self.max_sent_len = MAX_SENTENCE_LENGTH

        for sent in lang1_list:
            self.lang1_idx.append(sent)
            self.lang1_len.append(len(sent))

        for sent in lang2_list:
            self.lang2_idx.append(sent)
            self.lang2_len.append(len(sent))

        assert (len(self.lang1_idx) == len(self.lang1_len) == len(self.lang2_idx) == len(self.lang2_len))

    def __len__(self):
        return len(self.lang1_idx)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        return [self.lang1_idx[key], self.lang1_len[key], self.lang2_idx[key], self.lang2_len[key]]

    def vocab_collate_func(self, batch):
        """
        Customized function for DataLoader that dynamically pads the batch so that all
        data have the same length
        """
        batch = np.array(batch)

        lang1_idxs = batch[:, 0]
        lang1_lens = batch[:, 1]
        lang2_idxs = batch[:, 2]
        lang2_lens = batch[:, 3]

        #lang1_max = min(max(lang1_lens), max_sent_len )
        #lang2_max = min(max(lang2_lens), max_sent_len )

        pad_lang1_idxs = []
        pad_lang2_idxs = []
        #reset these so we used the max limit
        lang1_lens = []
        lang2_lens = []

        for sent in lang1_idxs:
            sent = sent[:self.max_sent_len]
            lang1_lens.append(len(sent))
            padded_vec = np.pad(np.array(sent),
                                pad_width=((0, self.max_sent_len - len(sent))),
                                mode="constant", constant_values=0)
            pad_lang1_idxs.append(padded_vec)

        for sent in lang2_idxs:
            sent = sent[:self.max_sent_len]
            lang2_lens.append(len(sent))
            padded_vec = np.pad(np.array(sent),
                                pad_width=((0, self.max_sent_len  - len(sent))),
                                mode="constant", constant_values=0)
            pad_lang2_idxs.append(padded_vec)

        ind_dec_order = np.argsort(lang1_lens)[::-1]

        pad_lang1_idxs = torch.from_numpy(np.array(pad_lang1_idxs)[ind_dec_order].astype(np.long)).long().to(device)
        pad_lang2_idxs = torch.from_numpy(np.array(pad_lang2_idxs)[ind_dec_order].astype(np.long)).long().to(device)
        lang1_lens = torch.from_numpy(np.array(lang1_lens)[ind_dec_order].astype(np.long))
        lang2_lens = torch.from_numpy(np.array(lang2_lens)[ind_dec_order].astype(np.long))
        return [pad_lang1_idxs, lang1_lens, pad_lang2_idxs, lang2_lens]



class EncoderRNN(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, num_layers, bidirectional):
        super(EncoderRNN, self).__init__()

        self.hidden_size = embedding_dim
        self.num_layers = num_layers
        self.directions = 1 + bidirectional
        self.bidirectional = bidirectional
        self.model_type = 'rnn'

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=Language.PAD_IDX)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[Language.PAD_IDX], 0)

        self.dropout_in = nn.Dropout(0.1)

        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=bidirectional, num_layers=self.num_layers)

    def forward(self, input, lengths):

        #batch_size, seq_len = input.size()
        #h1 = torch.randn(self.num_layers * self.multi, batch_size, self.hidden_size, device=device)
        #embedded = self.embedding(input)
        #output, hidden = self.gru(embedded, h1)

        batch_size, seq_len = input.shape

        h0, c0 = self.init_hidden(batch_size)

        # get embedding of characters
        word_embedded = self.embedding(input)
        word_embedded = self.dropout_in(word_embedded)
        # pack padded sequence

        word_embedded = torch.nn.utils.rnn.pack_padded_sequence(word_embedded, lengths.numpy(), batch_first=True)
        # fprop though RNN
        rnn_out, hidden = self.rnn(word_embedded, (h0, c0))

        # undo packing
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, total_length=seq_len)

        return rnn_out, hidden[0], hidden[1]

    def init_hidden(self, batch_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.zeros(self.num_layers * self.directions, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers * self.directions, batch_size, self.hidden_size).to(device)
        #if we need cell, depends on what type of rrn we're using
        return hidden, cell


class EncoderCNN(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, seqlen):
        super(EncoderCNN, self).__init__()

        self.hidden_size = embedding_dim
        #self.num_layers = num_layers
        #self.bidirectional = bidirectional
        #self.directions = 1 + bidirectional
        self.model_type = 'cnn'

        #if (num_layers != 1):
        #    print("Currently ignoring num_layers (",num_layers,") in ConvEncoder", sep='')

        self.word_embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=Language.PAD_IDX)
        self.pos_embedding = nn.Embedding(seqlen, embedding_dim, padding_idx=Language.PAD_IDX)

        #nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        #nn.init.constant_(self.embedding.weight[Language.PAD_IDX], 0)

        self.dropout_in = nn.Dropout(0.1)

        self.conv1a = nn.Conv1d(embedding_dim, embedding_dim*2, kernel_size=3, padding=1)
        self.conv1b = nn.Conv1d(embedding_dim*2, embedding_dim*2, kernel_size=3, padding=1)
        self.conv1c = nn.Conv1d(embedding_dim*2, embedding_dim*2, kernel_size=3, padding=1)
        self.conv1d = nn.Conv1d(embedding_dim * 2, embedding_dim * 2, kernel_size=3, padding=1)
        self.conv1e = nn.Conv1d(embedding_dim * 2, embedding_dim * 2, kernel_size=3, padding=1)

    def forward(self, input, lengths):

        batch_size, seq_len = input.shape

        pos = np.array(  [range(0,seq_len,1)] * batch_size , dtype=np.long)
        positions = torch.from_numpy(pos).long().to(device)

        # get embedding of characters and positions
        word_embedded = self.word_embedding(input)
        pos_embedded = self.pos_embedding(positions)

        embedded = self.dropout_in(word_embedded+pos_embedded)
        embedded = embedded.transpose(1,2)

        #print('in:',embedded.shape)
        output1 = torch.tanh(self.conv1a(embedded))
        output1 = torch.tanh(self.conv1b(output1) + output1)
        output1 = torch.tanh(self.conv1c(output1) + output1)
        output1 = torch.tanh(self.conv1d(output1) + output1)
        output1 = torch.tanh(self.conv1e(output1) + output1)

        output1 = output1.transpose(1, 2)
        #print('out:', output1.shape)

        #batch, seqlen, embeddim
        return output1, None , None


class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, vocab_size, num_layers=2, bidirectional=False):
        super(DecoderRNN, self).__init__()
        self.hidden_size = embedding_size
        self.context = 1 + (bidirectional or num_layers==2)
        self.directions = 1 + bidirectional
        self.num_layers = num_layers
        self.model_type = 'rnn'

        self.dropout_in = nn.Dropout(0.1)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=Language.PAD_IDX)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[Language.PAD_IDX], 0)

        self.rnn = nn.LSTM(embedding_size + (self.hidden_size * self.context), self.hidden_size, bidirectional=bidirectional, num_layers=self.num_layers)
        self.dropout_out = nn.Dropout(0.1)
        self.out = nn.Linear(self.hidden_size * self.directions, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    #take a dummy var and returned empty attention
    def forward(self, input, hidden, context, context2):

        word_embedded = self.embedding(input)
        word_embedded = self.dropout_in(word_embedded)
        full_input = torch.cat((word_embedded, context), dim=2)

        output, hidden = self.rnn(full_input, hidden)
        #print(output[0].shape)
        output = output.squeeze(0)  #get rid of the seqlen dim
        output = self.out(output)
        #print(output.shape)
        output = self.softmax(output)  #softmaxing across vocabsize
        return output, hidden, []

    def init_hidden(self, batch_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.zeros(self.num_layers * self.directions, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers * self.directions, batch_size, self.hidden_size).to(device)
        #if we need cell, depends on what type of rrn we're using
        return hidden, cell


#assumes bidirectional
#assumes layer x dir, batch, hidden
#returns layer, batch, hidden x dir
def combine_directions(hid):
    num_directions = 2
    num_layers, batch_size, hidden_size = hid.shape
    num_layers = int(num_layers / num_directions)
    #print('hid in :', hid.shape)
    hid = hid.view(num_layers, num_directions, batch_size, hidden_size)
    #print('hid view:', hid.shape)
    hid = hid.transpose(1,2).contiguous().view(num_layers, batch_size, -1)
    #print('hid transform:', hid.shape)
    return hid

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, vocab_size, num_layers=2,  dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()

        # Define parameters
        self.model_type = 'attn'
        self.hidden_size = embedding_size * 2
        self.directions = 1
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=Language.PAD_IDX)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = nn.Linear(self.hidden_size , self.hidden_size)

        self.attn_combine = nn.Linear(embedding_size +self.hidden_size , self.hidden_size)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers, dropout=dropout_p)

        #self.rnn = nn.LSTM(embedding_size + self.hidden_size, self.hidden_size, num_layers, dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, word_input, last_hidden, encoder_outputs, context2):

        # get the seqlen first
        encoder_outputs_seqfirst = encoder_outputs.transpose(0, 1)
        seq_len, batch_size, _ = encoder_outputs_seqfirst.shape

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input)
        word_embedded = self.dropout(word_embedded)

        #grap just the hidden from the lstm, and just the last layer
        hidden = last_hidden[0][-1]

        #if we don't have a seperate context, just reuse the encoder_outputs
        if context2 is None:
            context2 = encoder_outputs

        #print("attn_weights:", attn_weights.size())
        #print("word:", word_embedded.shape)
        #print("encoder (seqfirst) output:", encoder_outputs_seqfirst.shape)
        #print("last_hidden[0]:", last_hidden[0].shape)
        #print("hidden", hidden.shape)

        #"general" attention w = H . We
        We = self.attn(encoder_outputs)
        attn_weights = torch.bmm(We, hidden.unsqueeze(2))

        #get the seq dim in back
        attn_weights = attn_weights.transpose(1,2)

        #soft max across the sequences
        attn_weights = F.softmax(attn_weights,dim=2)
        attn_applied = torch.bmm(attn_weights, context2)

        #get seqlen back up front
        attn_applied = attn_applied.transpose(0,1)

        #create the input by concatinating word and attention context
        rnn_input = torch.cat((word_embedded, attn_applied), dim=2)

        rnn_input = torch.tanh(self.attn_combine(rnn_input))

        output, hidden = self.rnn(rnn_input, last_hidden)

        output = output.squeeze(0)  # get rid of the seqlen dim, we're always given just 1
        output = self.out(output)
        output = self.softmax(output)  # softmaxing across vocabsize

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

    def init_hidden(self, batch_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.zeros(self.num_layers * self.directions, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers * self.directions, batch_size, self.hidden_size).to(device)
        #if we need cell, depends on what type of rrn we're using
        return hidden, cell

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def save_model(encoder, decoder, save_prefix, label):
    fne = os.path.join(save_prefix, 'encoder_model_' + str(label) + '.st')
    fnd = os.path.join(save_prefix, 'decoder_model_' + str(label) + '.st')
    torch.save(encoder.state_dict(), fne)
    torch.save(decoder.state_dict(), fnd)

def load_model(model, model_path):

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)


def evaluate(encoder, decoder, sentences, lengths, beam=1):
    """
    Function that generate translation.
    First, feed the source sentence into the encoder and obtain the hidden states from encoder.
    Secondly, feed the hidden states into the decoder and unfold the outputs from the decoder.
    Lastly, for each outputs from the decoder, collect the corresponding words in the target language's vocabulary.
    And collect the attention for each output words.
    @param encoder: the encoder network
    @param decoder: the decoder network
    @param sentences: batch of sentences to be evaluated
    @param lengths: batch of sentence lengs
    @output decoded_words: a list of words in target language
    @output decoder_attentions: a list of vector, each of which sums up to 1.0
    """
    # process input sentence

    #assert torch_nograd is here

    batch_size, input_length = sentences.shape
    encoder_outputs, encoder_hidden, encoder_cell = encoder(sentences, lengths)
    context2 = None
    #we do the search one sentence at a time, so make it batch first
    #hid = encoder_hidden[0].transpose(0, 1)
    #cell = encoder_hidden[1].transpose(0, 1)

    if encoder.model_type == 'cnn':
        context = encoder_outputs
        context2, _, _ = encoder(sentences, lengths)
        decoder_hidden = decoder.init_hidden(batch_size)
    elif encoder.model_type == 'rnn':
        if encoder.bidirectional or encoder.num_layers == 2:
            context = torch.cat((encoder_hidden[-2], encoder_hidden[-1]), dim=1)
        else:
            context = encoder_hidden[-1]
        context = context.unsqueeze(0)
        context2 = context
        decoder_hidden = (encoder_hidden, encoder_cell)

    if decoder.model_type == 'attn' and encoder.model_type != 'cnn':
        context = encoder_outputs
        context2 = context
        encoder_hidden = combine_directions(encoder_hidden)
        encoder_cell = combine_directions(encoder_cell)
        # print('eh:', encoder_hidden.shape)
        decoder_hidden = (encoder_hidden, encoder_cell)

    all_decoded_words = []
    for i in range(batch_size):
        if (beam > 1):
            #decoded_words, decoder_attentions = beam_search(decoder, decoder_hidden, context[i].unsqueeze(0), context2[i].unsqueeze(0), input_length, beam)
            decoded_words, decoder_attentions = alternate_beam_search(decoder, decoder_hidden, context[i].unsqueeze(0),
                                                            context2[i].unsqueeze(0), input_length, beam)
        else:
            decoded_words, decoder_attentions = greedy_search(decoder, decoder_hidden, context[i].unsqueeze(0), context2[i].unsqueeze(0), input_length)
        all_decoded_words.append(decoded_words)

    return all_decoded_words, decoder_attentions#[:di + 1]


def greedy_search(decoder, decoder_hidden, context, context2, max_length):

    #print("greedy context:", context.shape)

    batch_size = context.shape[0]
    decoder_input = torch.tensor([Language.SOS_IDX] * batch_size, device=device)  # SOS
    # output of this function
    decoded_words = []
    decoder_attentions = torch.zeros(max_length*2, batch_size, max_length)
    for di in range(max_length):
         decoder_input = decoder_input.unsqueeze(0)
         # for each time step, the decoder network takes two inputs: previous outputs and the previous hidden states
         decoder_output, decoder_hidden, decoder_attention = decoder(
             decoder_input, decoder_hidden, context, context2)

         topv, topi = decoder_output.topk(1)
         decoder_input = topi.squeeze().detach().unsqueeze(0)
         decoded_words.append(decoder_input.item())
         if (len(decoder_attention)> 0):
             #print("d_a:",decoder_attention.shape)
             decoder_attentions[di] = decoder_attention
         if topi == Language.EOS_IDX:
             break

    return decoded_words, decoder_attentions


def single_beam_search(decoder, decoder_hidden, context, max_length = 100, beam_width=3):

    decoder_input = torch.tensor(beam_width, device=device).fill(Language.SOS_IDX)
    decoder_attentions = torch.zeros(max_length*2, batch_size, max_length)
    hypotheses = []
    candidates = []
    scores = []
    for index in range(beam_width * 2):
        candidates[index] = []
        scores[index] = 0
        if index % 2 == 0:
            hypotheses[index/2].append(torch.tensor(Language.SOS_IDX))      

    for step in range(max_length):
        for hyp in hypotheses:
            if hyp[-1] == Language.EOS_IDX:
                next
            else:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(beam_width)
                possible[topv] = topi
                best_val = sorted(possible.keys)[:beam_width]
                beam_candidates = [(v, possible[v]) for v in best_val]
        

        
        
        

    return decoder_attentions

#Not working, multibatch beam search is hard
def beam_search(decoder, decoder_hidden, context, context2, max_length = 100, beam_width=2):

    batch_size = context.shape[0]
    decoder_input = torch.tensor(batch_size, beam_width, device=device).fill(Language.SOS_IDX)
    decoder_attentions = torch.zeros(max_length*2, batch_size, max_length)
    
    beam = Beam(Language, device)
    scores = None

    for step_num in range(max_length):
        
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, 
                                                                    decoder_hidden, 
                                                                    encoder_outputs)

        scores, indices, hypothesis = beam.step(step_num, logprobs, scores)
        
        

    return decoder_attentions



class BeamPart:

    def __init__(self, token, decoder_hidden):

        self.decoded_word_list = []
        self.prob = 1
        self.decoded_word_list.append(token)
        self.decoder_hidden = decoder_hidden

def alternate_beam_search(decoder, decoder_hidden, context, context2, max_length, beam_width):

    batch_size = context.shape[0]
    decoder_input = torch.tensor(Language.SOS_IDX , device=device)  # SOS
    # output of this function

    #initialize our beams
    beams = []
    for i in range(beam_width):
        beams.append(BeamPart(decoder_input, decoder_hidden ))

    #only go max_length words, even if our best beam is still going
    for di in range(max_length):

        candidates = []
        for b in range(beam_width):
            beam = beams[b]

            #don't add anything if we've already hit EOS
            if beam.decoded_word_list[-1] == Language.EOS_IDX:
                candidates.append([beam, beam.prob, Language.EOS_IDX])
            else:
                decoder_input = beam.decoded_word_list[-1].unsqueeze(0).unsqueeze(0)
                # for each time step, the decoder network takes two inputs: previous outputs and the previous hidden states
                #print(decoder_input.shape, beam.decoder_hidden[0].shape, beam.decoder_hidden[1].shape, context.shape, context2.shape)
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, beam.decoder_hidden, context, context2)

                #print('do:', decoder_output)
                topv, topi = decoder_output.topk(beam_width) # we can't need to look at more than these
                topv = topv.cpu().numpy()[0]
                topv = np.exp(topv) # to reverse the sort
                topi = topi[0]
                beam.decoder_hidden = decoder_hidden

                for b2 in range(beam_width):
                    candidates.append([beam, topv[b2] * beam.prob, topi[b2]])

        sorted(candidates, key=lambda x: x[1], reverse=True)

        beams = []
        for b in range(beam_width):
            beam = copy.deepcopy(candidates[b][0])
            beam.prob = -1 * candidates[b][1] #reverse it back
            if beam.decoded_word_list[-1] != Language.EOS_IDX:
                beam.decoded_word_list.append(candidates[b][2])
            beams.append(beam)

    #since we just want one, faster to scan than to resort I think
    best_prob = -1
    best_beam = None
    for beam in beams:
        if (beam.prob > best_prob):
            best_beam = beam
            best_prob = beam.prob

    decoded_words = best_beam.decoded_word_list[1:]

    return decoded_words, None

def evaluateRandomly(lang1, lang2, input_lang, output_lang, encoder, decoder, n=10):
    """
    Randomly select a English sentence from the dataset and try to produce its French translation.
    Note that you need a correct implementation of evaluate() in order to make this function work.
    """
    for i in range(n):
        idx = random.choice(range(len(lang1)))
        print('>', ' '.join([input_lang.index2word[x] for x in lang1[idx]]))
        print('=', ' '.join([output_lang.index2word[x] for x in lang2[idx]]))
        #greedy is beam=-1
        output_words, attentions = evaluate(encoder, decoder, torch.from_numpy(np.array(lang1[idx])).unsqueeze(0), torch.from_numpy(np.array(len(lang1[idx]))).unsqueeze(0), -1)
        output_words = [output_lang.index2word[x] for x in output_words]
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def evaluateBLUE(lang1, lang2, output_lang, encoder, decoder,  max_sent_len, beam_width=1 ):
    """
    Randomly select a English sentence from the dataset and try to produce its French translation.
    Note that you need a correct implementation of evaluate() in order to make this function work.
    """
    list_of_references = []
    list_of_hypotheses = []

    for i in tqdm.tqdm(range(len(lang1))):
        #TODO: should probably just use a loader here
        l1 = lang1[i][:max_sent_len]
        pad_lang1 = np.pad(np.array(l1),
               pad_width=((0, max_sent_len - len(l1))),
               mode="constant", constant_values=0)
        #print("leng of input:", len(pad_lang1), len(l1))
        target_words = [output_lang.index2word[x] for x in lang2[i]]
        list_of_references.append([target_words])
        output_tokens, attentions = evaluate(encoder, decoder, torch.tensor(pad_lang1, device=device).long().unsqueeze(0), torch.tensor(len(l1)).unsqueeze(0), beam_width)
        output_words = [output_lang.index2word[x] for x in output_tokens[0]]
        list_of_hypotheses.append(output_words)

        if(random.random() < 0.005):
            print('tw:"',' '.join(target_words), '"')
            print('ow:"',' '.join(output_words), '"')

        chencherry = nltk.bleu_score.SmoothingFunction()

    return nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypotheses, smoothing_function=chencherry.method1)