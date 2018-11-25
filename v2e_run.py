import torch
import translator
import pickle

#should make these paramaters

#define our model and training run paramaters
hidden_size = 512
iters = 100000
print_int = 5000
plot_int = 1000
lr = 0.01
max_length = 30
max_vocab = -1  #not used yet
teacher_forcing_ratio = 0.5

#need to run these both to avoid passing device around everywhere
translator.init_device()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#loads the data from the two files and creates two vocabs and a set of string sentence tuples
input_vocab, output_vocab, pairs = translator.loadData('data/iwslt-vi-en/train.tok.vi', 'data/iwslt-vi-en/train.tok.en')

#tokenize and convert to tensors
pairs = [translator.tensorsFromPair(pair, input_vocab, output_vocab) for pair in pairs]


#define our encoder and decoder
encoder1 = translator.EncoderRNN(input_vocab.n_words, hidden_size).to(device)
#attn_decoder1 =translator.DecoderRNN(hidden_size, output_lang.n_words).to(device)
attn_decoder1 = translator.BahdanauAttnDecoderRNN(hidden_size, output_vocab.n_words, n_layers=1, dropout_p=0.1).to(device)


#pairs needs to be a array of tuples of input_tokens, output_tokens
plot_losses = translator.trainIters(pairs, encoder1, attn_decoder1, iters, max_length, teacher_forcing_ratio, lr,
                                    input_vocab, output_vocab, print_every=print_int, plot_every=plot_int, save_every=print_int)

pickle.dump(plot_losses, open('data/losses.p', 'wb'))

torch.save(encoder1.state_dict(), 'data/final_encoder_model.st')
torch.save(attn_decoder1.state_dict(), 'data/final_decoder_model.st')
