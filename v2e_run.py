import torch
import translator
import pickle
from argparse import ArgumentParser



if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("-hs", "--hidden_size", type=int, default = 512,
                    help="hidden size dimmension")

    ap.add_argument("-it", "--iterations", type=int, default = 200000,
                    help="")

    ap.add_argument("-pr", "--print_iter", type=int, default=5000,
                    help="")

    ap.add_argument("-pl", "--plot_iter", type=int, default=1000,
                    help="")

    ap.add_argument("-lr", "--learning_rate", type=float, default=0.01,
                    help="")

    ap.add_argument("-ml", "--max_length", type=int, default=50,
                    help="")

    ap.add_argument("-mv", "--max_vocab", type=int, default=-1,
                    help="")

    ap.add_argument("-tf", "--teacher_force", type=float, default=0.5,
                    help="")

    ap.add_argument("-sl", "--source_file", default='data/iwslt-vi-en/train.tok.vi',
                    help="")

    ap.add_argument("-tl", "--target_file", default='data/iwslt-vi-en/train.tok.en',
                    help="")

    args = vars(ap.parse_args())

    #define our model and training run paramaters
    hidden_size = args['hidden_size']
    iters = args['iterations']
    print_int = args['print_iter']
    plot_int = args['plot_iter']
    lr = args['learning_rate']
    max_length = args['max_length']
    max_vocab = args['max_vocab']  #not used yet
    teacher_forcing_ratio = args['teacher_force']

    lang1_file = args['source_file']
    lang2_file = args['target_file']

    #need to run these both to avoid passing device around everywhere
    translator.init_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #loads the data from the two files and creates two vocabs and a set of string sentence tuples
    input_vocab, output_vocab, pairs = translator.loadData(lang1_file, lang2_file)

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
