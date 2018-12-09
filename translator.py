from io import open
import unicodedata
import re
import random
import time
import math
import nltk

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import os
import numpy as np
from collections import Counter


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
        self.emodel_type = 'rnn'

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
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        return rnn_out, hidden[0], hidden[1]

    def init_hidden(self, batch_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.zeros(self.num_layers * self.directions, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers * self.directions, batch_size, self.hidden_size).to(device)
        #if we need cell, depends on what type of rrn we're using
        return hidden, cell


class EncoderCNN(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, seqlen,  num_layers, bidirectional):
        super(EncoderCNN, self).__init__()

        self.hidden_size = embedding_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 1 + bidirectional
        self.emodel_type = 'cnn'

        if (num_layers != 1):
            print("Currently ignoring num_layers (",num_layers,") in ConvEncoder", sep='')


        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=Language.PAD_IDX)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[Language.PAD_IDX], 0)

        self.dropout_in = nn.Dropout(0.1)

        self.conv = nn.Conv1d(seqlen, self.directions, kernel_size=3, padding=1)

    def forward(self, input, lengths):

        #batch_size, seq_len = input.size()
        #h1 = torch.randn(self.num_layers * self.multi, batch_size, self.hidden_size, device=device)
        #embedded = self.embedding(input)
        #output, hidden = self.gru(embedded, h1)

        batch_size, seq_len = input.shape

        # get embedding of characters
        word_embedded = self.embedding(input)
        word_embedded = self.dropout_in(word_embedded)
        # pack padded sequence
        print('in:',word_embedded.shape)
        output = self.conv(word_embedded)
        print ('out:',output.shape)

        if self.bidirectional:
            output = output.view(batch_size, self.directions * self.hidden_size)

        return output, None, None




class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, vocab_size, num_layers=2, bidirectional=True):
        super(DecoderRNN, self).__init__()
        self.hidden_size = embedding_size
        self.directions = 1 + bidirectional
        self.num_layers = num_layers

        self.dropout_in = nn.Dropout(0.1)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=Language.PAD_IDX)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[Language.PAD_IDX], 0)

        self.rnn = nn.LSTM(embedding_size + (self.hidden_size * self.directions), self.hidden_size, bidirectional=bidirectional, num_layers=self.num_layers)
        self.dropout_out = nn.Dropout(0.1)
        self.out = nn.Linear(self.hidden_size * self.directions, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    #take a dummy var and returned empty attention
    def forward(self, input, hidden, context, encoder_outputs):

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

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()

        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = nn.Linear(hidden_size*2, hidden_size)
        if n_layers == 1:
            dropout_p = 0  #can't dropout with just 1 layer

        self.rnn = nn.LSTM(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note that we will only be running forward for a single decoder time step, but will use all encoder outputs

        #print('word_input', word_input.size(), word_input, flush=True)

        # Get the embedding of the current input word (last output word)
        _, batch_size = word_input.shape
        word_embedded = self.embedding(word_input)#.view(1, batch_size, -1)  # S=1 x B x N
        word_embedded = self.dropout(word_embedded)

        bs, seq_len, _ = encoder_outputs.shape

        attn_weights = torch.zeros((bs,seq_len)).to(device)

        # Calculate energies for each encoder output
        print('hidden:', last_hidden[0].shape)
        print('attn weights:', attn_weights.shape)
        print('encoder out:', encoder_outputs.shape)
        print('word_embedded', word_embedded.shape)

        #probably should work in the cell here too?
        attn_weights = F.softmax(
            self.attn(torch.cat((word_embedded[0], last_hidden[0][0]), 1)), dim=1)

        print('attn_weights', attn_weights.shape)

        #for i in range(seq_len):

         #   a  = self.attn(encoder_outputs[:,i])
         #   print('a:', a.shape)
         #   at = last_hidden[0].squeeze(0).dot(a)

         #   attn_weights[i] = F.softmax(at, dim=0).to(device)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0).unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        rnn_input = torch.cat((word_embedded[0], attn_applied[0]), 1)
        rnn_input = F.relu(rnn_input).unsqueeze(0)

        output, hidden = self.rnn(rnn_input, last_hidden)

        # Final output layer
        output = output.squeeze(0)  # B x N
        output = F.log_softmax(self.out(output), dim=1)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights



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

def save_loses(plot_losses, save_prefix):
    fn = os.path.join(save_prefix, 'losses.p')
    pickle.dump(plot_losses, open(fn, 'wb'))


def save_model(encoder, decoder, save_prefix, label):
    fne = os.path.join(save_prefix, 'encoder_model_' + label + '.st')
    fnd = os.path.join(save_prefix, 'decoder_model_' + label + '.st')
    torch.save(encoder.state_dict(), fne)
    torch.save(decoder.state_dict(), fnd)


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluate(encoder, decoder, sentences, lengths, beam=0):
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

    #we do the search one sentence at a time, so make it batch first
    hid = encoder_hidden[0].transpose(0, 1)
    cell = encoder_hidden[1].transpose(0, 1)

    if encoder.emodel_type == 'cnn':
        context = encoder_outputs.squeeze(1)
        decoder_hidden = decoder.init_hidden(batch_size)
    elif encoder.emodel_type == 'rnn':
        if encoder.bidirectional:
            context = torch.cat((encoder_hidden[-2], encoder_hidden[-1]), dim=1)
        else:
            context = encoder_hidden[-1]

        decoder_hidden = (encoder_hidden, encoder_cell)

    all_decoded_words = []
    for i in range(hid.shape[0]):
        if (beam > 0):
            decoded_words, decoder_attentions = beam_search(decoder, (hid[i].unsqueeze(1), cell[i].unsqueeze(1)), encoder_outputs[i].unsqueeze(0), beam)
        else:
            decoded_words, decoder_attentions = greedy_search(decoder, (hid[i].unsqueeze(1), cell[i].unsqueeze(1)), context.unsqueeze(0), encoder_outputs[i].unsqueeze(0))
        all_decoded_words.append(decoded_words)

    return all_decoded_words, decoder_attentions#[:di + 1]


def greedy_search(decoder, decoder_hidden, context, encoder_outputs):

    max_length = 100
    batch_size = encoder_outputs.shape[0]
    decoder_input = torch.tensor([Language.SOS_IDX] * batch_size, device=device)  # SOS
    # output of this function
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    for di in range(max_length):
         decoder_input = decoder_input.unsqueeze(0)
         # for each time step, the decoder network takes two inputs: previous outputs and the previous hidden states
         decoder_output, decoder_hidden, decoder_attention = decoder(
             decoder_input, decoder_hidden, context, encoder_outputs)

         #TODO: implement beam search
         topv, topi = decoder_output.topk(1)
         decoder_input = topi.squeeze().detach().unsqueeze(0)
         decoded_words.append(decoder_input.item())
         if (len(decoder_attention)> 0):
             decoder_attentions[di] = decoder_attention
         if topi == Language.EOS_IDX:
             break

    return decoded_words, decoder_attentions

def beam_search(decoder, decoder_hidden ,encoder_outputs, beam_width=2):

    max_length = 100
    decoder_input = torch.tensor([[Language.SOS_IDX]], device=device)  # SOS
    # output of this function
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    beam_candidates = [Language.SOS_IDX]

    for step in range(max_length):
        possible = {}
        for cand in beam_candidates:
            if cand[1] == Language.EOS_IDX:
                possible[cand[0]] = cand[1]
            else:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(beam_width)
                possible[topv] = topi
            best_val = sorted(possible.keys)[:beam_width]
            beam_candidates = [(v, possible[v]) for v in best_val]

    return decoded_words, decoder_attentions


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

def evaluateBLUE(lang1, lang2, input_lang, output_lang, encoder, decoder):
    """
    Randomly select a English sentence from the dataset and try to produce its French translation.
    Note that you need a correct implementation of evaluate() in order to make this function work.
    """
    list_of_references = []
    list_of_hypotheses = []

    for i in range(len(lang1)):

        target_words = [output_lang.index2word[x] for x in lang2[i]]
        list_of_references.append([target_words])
        output_tokens, attentions = evaluate(encoder, decoder, torch.tensor(lang1[i], device=device).unsqueeze(0), torch.tensor(len(lang1[i])).unsqueeze(0), -1)
        output_words = [output_lang.index2word[x] for x in output_tokens[0]]
        list_of_hypotheses.append(output_words)

        if(random.random() < 0.005):
            print('tw:"',' '.join(target_words), '"')
            print('ow:"',' '.join(output_words), '"')

        chencherry = nltk.bleu_score.SmoothingFunction()

    return nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypotheses, smoothing_function=chencherry.method1)