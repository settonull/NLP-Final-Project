from io import open
import unicodedata
import re
import random
import time
import math
import nltk

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

'''
DS-GA 1011 Final Project
Seq2Seq With Attention Translatoion Model

Much code adapted from Lab8 and Pytorch.org examples.
'''


#doesn't actually get called I don't think ?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#a bit of a hack, this must be called first so all methods have access to our device
def init_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Lang:

    SOS_token = 0
    EOS_token = 1

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...", flush=True)

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def loadData(lang_file1, lang_file2):

    print("Loading data from ", lang_file1, "and", lang_file2,"...", flush=True)
    lang1 = open(lang_file1, 'r', encoding='utf-8')
    lang2 = open(lang_file2, 'r', encoding='utf-8')

    input_lang = Lang(lang_file1)
    output_lang = Lang(lang_file2)
    pairs = []
    #take 1, assumes we don't have unreasonable vocabs
    for sent1 in lang1:
        sent2 = lang2.readline()
        sent1 = sent1.strip()
        sent2 = sent2.strip()
        input_lang.addSentence(sent1)
        output_lang.addSentence(sent2)
        pairs.append((sent1, sent2))

    print("Read %s sentence pairs" % len(pairs), flush=True)
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words, flush=True)

    return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    #take a dummy var and returned empty attention
    def forward(self, input, hidden, dummy):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden, []

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


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
        self.attn = nn.Linear(hidden_size, hidden_size)
        if n_layers == 1:
            dropout_p = 0  #can't dropout with just 1 layer
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note that we will only be running forward for a single decoder time step, but will use all encoder outputs

        #print('word_input', word_input.size(), word_input, flush=True)

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1)  # S=1 x B x N
        word_embedded = self.dropout(word_embedded)

        seq_len = len(encoder_outputs)

        attn_weights = torch.zeros(seq_len).to(device)

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_weights[i] = F.softmax(last_hidden.squeeze(0).squeeze(0).dot(self.attn(encoder_outputs[i])), dim=0).to(device)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0).unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        rnn_input = torch.cat((word_embedded[0], attn_applied[0]), 1)
        rnn_input = F.relu(rnn_input).unsqueeze(0)

        output, hidden = self.gru(rnn_input, last_hidden)

        # Final output layer
        output = output.squeeze(0)  # B x N
        output = F.log_softmax(self.out(output), dim=1)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(Lang.EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)



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


def train(input_tensor, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, max_length, teacher_forcing_ratio):

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = min(input_tensor.size(0), max_length)
    target_length = min(target_tensor.size(0), max_length)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([Lang.SOS_token], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == Lang.EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(pairs, encoder, decoder, n_iters, max_length, teacher_forcing_ratio, learning_rate,
               input_vocab, output_vocab, print_every=1000, plot_every=100, save_every=-1):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = random.choices(pairs, k=n_iters)

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, max_length, teacher_forcing_ratio)
        print_loss_total += loss
        plot_loss_total += loss

        if print_every  > -1 & iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg), flush=True)
            b_start = time.time()
            b_num = int(n_iters/10)
            b = evaluateBLUE(pairs, max_length, input_vocab, output_vocab, encoder, decoder, b_num )
            b_end = time.time()
            print("Evaluated BLUE from sample of", b_num , ", result:", b * 100, "in", asMinutes(b_end - b_start), flush=True)

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if save_every > 0 and iter % save_every == 0:
            torch.save(encoder.state_dict(), 'data/encoder_model_' + str(iter) + '.st')
            torch.save(decoder.state_dict(), 'data/decoder_model_' + str(iter) + '.st')

    return plot_losses




def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluate(encoder, decoder, sentence, max_length, input_lang):
    """
    Function that generate translation.
    First, feed the source sentence into the encoder and obtain the hidden states from encoder.
    Secondly, feed the hidden states into the decoder and unfold the outputs from the decoder.
    Lastly, for each outputs from the decoder, collect the corresponding words in the target language's vocabulary.
    And collect the attention for each output words.
    @param encoder: the encoder network
    @param decoder: the decoder network
    @param sentence: string, a sentence in source language to be translated
    @param max_length: the max # of words that the decoder can return
    @output decoded_words: a list of words in target language
    @output decoder_attentions: a list of vector, each of which sums up to 1.0
    """
    # process input sentence
    with torch.no_grad():
        if type(sentence) == str:
            input_tensor = tensorFromSentence(input_lang, sentence)
            input_length = min(input_tensor.size()[0], max_length)
        else:
            input_tensor = sentence
            input_length = min(len(sentence), max_length)

        # encode the source lanugage
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[Lang.SOS_token]], device=device)  # SOS
        # decode the context vector
        decoder_hidden = encoder_hidden  # decoder starts from the last encoding sentence
        # output of this function
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            # for each time step, the decoder network takes two inputs: previous outputs and the previous hidden states
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            #This should be the greed implementation
            #need to do test, and implement beam
            topv, topi = decoder_output.topk(1)
            if topi == Lang.EOS_token:
                break
            decoder_input = topi.squeeze().detach()
            decoded_words.append(decoder_input.item())
            if (len(decoder_attention)> 0):
                decoder_attentions[di] = decoder_attention

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(pairs, max_length, input_lang, output_lang, encoder, decoder, n=10):
    """
    Randomly select a English sentence from the dataset and try to produce its French translation.
    Note that you need a correct implementation of evaluate() in order to make this function work.
    """
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], max_length, input_lang)
        output_words = [output_lang.index2word[x] for x in output_words]
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def evaluateBLUE(pairs, max_length, input_lang, output_lang, encoder, decoder, num):
    """
    Randomly select a English sentence from the dataset and try to produce its French translation.
    Note that you need a correct implementation of evaluate() in order to make this function work.
    """
    list_of_references = []
    hypotheses = []

    examples = list(range(num))
    random.shuffle(examples)

    for i in examples:
        pair = pairs[i]

        if (type(pair[0]) == str):
            input_words = pair[0].split(' ')
        else:
            input_words = [input_lang.index2word[x.item()] for x in pair[0]]

        list_of_references.append([input_words])
        output_words, attentions = evaluate(encoder, decoder, pair[0], max_length, input_lang)
        output_words = [output_lang.index2word[x] for x in output_words]
        hypotheses.append(output_words)

    return nltk.translate.bleu_score.corpus_bleu(list_of_references, hypotheses)