import torch
import translator
from argparse import ArgumentParser
import os
from torch import optim
import time
import torch.nn as nn
import random
import numpy as np

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("-hs", "--hidden_size", type=int, default = 512,
                    help="hidden size dimmension")
    ap.add_argument("-ed", "--embed_dim", type=int, default=512,
                    help="embedding dimmension")
    ap.add_argument("-it", "--epochs", type=int, default = 20,
                    help="")
    ap.add_argument("-pr", "--print_iter", type=int, default=5000,
                    help="")
    ap.add_argument("-pl", "--plot_iter", type=int, default=1000,
                    help="")
    ap.add_argument("-lr", "--learning_rate", type=float, default=0.01,
                    help="")
    ap.add_argument("-ml", "--max_length", type=int, default=50,
                    help="")
    ap.add_argument("-mv", "--max_vocab", type=int, default=45000,
                    help="")
    ap.add_argument("-tf", "--teacher_force", type=float, default=0.5,
                    help="")
    ap.add_argument("-sl", "--source_dir", default='data/iwslt-vi-en/',
                    help="")
    ap.add_argument("-tl", "--target_pair", default='vi-en',
                    help="")
    ap.add_argument("-dt", "--decoder_model_type", default='rnn',
                    help="rnn, attn")
    ap.add_argument("-et", "--encoder_model_type", default='rnn',
                    help="rnn, cnn")
    ap.add_argument("-md", "--model_directory", default='models',
                    help="where to put our models")
    ap.add_argument("-bs", "--batch_size", type=int, default=32,
                    help="")
    ap.add_argument("-nl", "--num_layers", type=int, default=2,
                    help="")
    ap.add_argument("-sd", "--single_direction", action="store_true",
                    help="Default to bidirectional, this turns it off")

    #TODO:learning rate schedule
    #TODO:loss annelaing

    args = vars(ap.parse_args())

    #define our model and training run paramaters
    hidden_size = args['hidden_size']
    embed_dim = args['embed_dim']
    epochs = args['epochs']
    print_every = args['print_iter']
    plot_every = args['plot_iter']
    learning_rate = args['learning_rate']
    max_length = args['max_length']
    max_vocab = args['max_vocab']  #not used yet
    teacher_forcing_ratio = args['teacher_force']
    batch_size = args['batch_size']

    lang_dir = args['source_dir']
    lang1, lang2 = args['target_pair'].split('-')
    dmodel_type = args['decoder_model_type']
    emodel_type = args['encoder_model_type']

    num_layers = args['num_layers']
    bidirectional = not args['single_direction']

    DEBUG = True

    #hack, use extension of first file as language type
    lang_label = lang1

    #need to run these both to avoid passing device around everywhere
    translator.init_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loads the data from the two files and creates two vocabs and a set of string sentence tuples
    input_vocab, train_input_sentences, val_input_sentences, test_input_sentences = translator.loadData(lang_dir, lang1, max_vocab)
    output_vocab, train_output_sentences , val_output_sentences , test_output_sentences = translator.loadData(lang_dir, lang2, max_vocab)

    train_input_index = translator.indexSentences(input_vocab, train_input_sentences)
    train_output_index = translator.indexSentences(output_vocab, train_output_sentences)

    val_input_index = translator.indexSentences(input_vocab, val_input_sentences)
    val_output_index = translator.indexSentences(output_vocab, val_output_sentences)

    train_dataset = translator.PairsDataset(train_input_index, train_output_index, max_length)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=train_dataset.vocab_collate_func, shuffle=True)

    val_dataset = translator.PairsDataset(val_input_index, val_output_index, max_length)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, collate_fn=val_dataset.vocab_collate_func)

    #Make sure indexed sentences are correct
    if DEBUG:
        rnd = random.randint(0,len(train_input_index)-1)
        print(lang1, 'index check')
        print(train_input_sentences[rnd])
        print([input_vocab.index2word[x] for x in train_input_index[rnd]])

        rnd = random.randint(0, len(train_output_index) - 1)
        print(lang2, 'index check')
        print(train_output_sentences[rnd])
        print([output_vocab.index2word[x] for x in train_output_index[rnd]])
        print()

        print(output_vocab.index2word[0:10])

    # define our encoder and decoder
    encoder = None
    decoder = None

    if emodel_type == 'rnn':
        encoder = translator.EncoderRNN(input_vocab.n_words, embed_dim, num_layers, bidirectional).to(device)
    elif emodel_type == 'cnn':
        encoder = translator.EncoderCNN(input_vocab.n_words, embed_dim, max_length, num_layers, bidirectional).to(device)
    else:
        print("unknown model_type", emodel_type)
        exit(1)

    decoder = None

    if dmodel_type == 'rnn':
        decoder =translator.DecoderRNN(hidden_size, output_vocab.n_words, num_layers, bidirectional).to(device)
    elif dmodel_type == 'attn':
        decoder = translator.BahdanauAttnDecoderRNN(hidden_size, output_vocab.n_words, n_layers=1, dropout_p=0.1).to(device)
    else: #TODO: implement other model types
        print("unknown model_type", dmodel_type)
        exit(1)

    save_prefix = os.path.join(args['model_directory'], lang_label, emodel_type + '-' + dmodel_type)
    os.makedirs(save_prefix, exist_ok=True)
    print("Running on ", device)
    print("Using model types:", emodel_type, dmodel_type)
    print("saving in:", save_prefix, flush=True)
    print("Layers:", num_layers, "bidirectionl" if bidirectional else "")
    print()

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss(ignore_index=translator.Language.PAD_IDX)

    #blu = translator.evaluateBLUE(val_input_index, val_output_index, input_vocab, output_vocab, encoder, decoder)
    #print("Val Blue:", blu)

    print("Begin Training!", flush=True)
    plot_losses = []
    start = time.time()
    for epoch in range(epochs):

        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        total_batches = len(train_loader)

        encoder.train()
        decoder.train()

        for i, (lang1, lengths1, lang2, lengths2) in enumerate(train_loader):
            #print("epoch, batch", epoch, 'of', epochs, i, 'of' , total_batches)
            batch_size, input_length = lang1.shape
            batch_size, target_length = lang2.shape

            target_tensor = lang2.transpose(0,1)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss = 0

            encoder_outputs, encoder_hidden, encoder_cell= encoder(lang1, lengths1)
            #print(encoder_hidden.shape)

            if emodel_type == 'cnn':
                context = encoder_outputs.squeeze(1)
                decoder_hidden = decoder.init_hidden(batch_size)
            elif emodel_type == 'rnn':
                if bidirectional:
                    context = torch.cat((encoder_hidden[-2], encoder_hidden[-1]), dim=1)
                else:
                    context = encoder_hidden[-1]

                decoder_hidden = (encoder_hidden, encoder_cell)

            #print('context:', context.shape)

            #make this 1 x batchsize
            decoder_input = torch.tensor([translator.Language.SOS_IDX] * batch_size, device=device)

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            debug_decode = []

            decoder_full_out = torch.zeros(batch_size,target_length,output_vocab.n_words, device=device)

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_input = decoder_input.unsqueeze(0)
                    #print("TF:", decoder_input.shape)
                    #print("TFc:", context.unsqueeze(0).shape)
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, context.unsqueeze(0), encoder_outputs)
                    decoder_full_out[:, di] = decoder_output

                    decoder_input = target_tensor[di]  # Teacher forcing
            else:
                #rnd  = random.randint(0, len(lang2)-1)
                #if (DEBUG) & (print_every > -1) & (i % print_every == 0) & (i > 0):
                #    print('in:', lang2[rnd].tolist())

                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_input = decoder_input.unsqueeze(0)
                    #print("SL:", decoder_input.shape)
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, context.unsqueeze(0), encoder_outputs)
                    topv, topi = decoder_output.topk(1)
                    #print("do:",decoder_output.shape)
                    #print("ti:",target_tensor[di].shape)
                    decoder_full_out[:, di] = decoder_output
                    #print('l:',loss)
                    decoder_input = topi.squeeze().detach()  # detach from history as input
                    #debug_decode.append(decoder_input[rnd].item())
                #if (DEBUG) & (print_every > -1) & (i % print_every == 0) & (i > 0):
                ##   print('out:', debug_decode)


            decoder_full_out = decoder_full_out.transpose(1, 2)
            #print(decoder_full_out.shape, lang2.shape)
            loss = criterion(decoder_full_out, lang2)
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            loss = loss.item()

            print_loss_total += loss
            plot_loss_total += loss

            if (print_every > -1) & (i % print_every == 0) & (i > 0):
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('epoch %d : %s (%d %d%%) Loss: %.4f' %  (epoch, translator.timeSince(start, i / total_batches),
                                             i, i / total_batches * 100, print_loss_avg), flush=True)

            if i % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            #TODO: Save model someplace
            #if save_every > 0 and iter % save_every == 0:
            #    save_model(encoder, decoder, save_prefix, str(iter))

        print("Computing VAL", flush=True)

        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            val_loss = 0
            for i, (lang1, lengths1, lang2, lengths2) in enumerate(val_loader):

                # print("epoch, batch", epoch, 'of', epochs, i, 'of' , total_batches)
                batch_size, input_length = lang1.shape
                batch_size, target_length = lang2.shape

                target_tensor = lang2.transpose(0, 1)

                encoder_outputs, encoder_hidden, encoder_cell = encoder(lang1, lengths1)

                if emodel_type == 'cnn':
                    context = encoder_outputs.squeeze(1)
                    decoder_hidden = decoder.init_hidden(batch_size)
                elif emodel_type == 'rnn':
                    if bidirectional:
                        context = torch.cat((encoder_hidden[-2], encoder_hidden[-1]), dim=1)
                    else:
                        context = encoder_hidden[-1]

                    decoder_hidden = (encoder_hidden, encoder_cell)

                # make this 1 x batchsize
                decoder_input = torch.tensor([translator.Language.SOS_IDX] * batch_size, device=device)

                decoder_hidden = encoder_hidden

                decoder_full_out = torch.zeros(batch_size, target_length, output_vocab.n_words, device=device)

                for di in range(target_length):
                    decoder_input = decoder_input.unsqueeze(0)
                    # print("SL:", decoder_input.shape)
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, context.unsqueeze(0), encoder_outputs)
                    topv, topi = decoder_output.topk(1)
                    decoder_full_out[:, di] = decoder_output
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                decoder_full_out = decoder_full_out.transpose(1, 2)
                # print(decoder_full_out.shape, lang2.shape)
                loss = criterion(decoder_full_out, lang2)
                # print(loss)
                val_loss += loss.item()

            print("Val loss:", val_loss / len(val_loader), flush=True)
            blu = translator.evaluateBLUE(val_input_index, val_output_index, input_vocab, output_vocab, encoder, decoder)
            print("Val Blue:", blu, flush=True)

