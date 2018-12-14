import os
import time
import random
from argparse import ArgumentParser

import torch
import torch.optim as optim
import torch.nn as nn

import translator


#evaluate the model, given the trained encoder and decoder, and a dataset and loss criterion
def eval_model(encoder, decoder, val_loader, criterion):

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        val_loss = 0
        for i, (lang1, lengths1, lang2, lengths2) in enumerate(val_loader):

            batch_size, input_length = lang1.shape
            batch_size, target_length = lang2.shape

            encoder_outputs, encoder_hidden, encoder_cell = encoder(lang1, lengths1)
            context2 = None

            if encoder.model_type == 'cnn':
                context = encoder_outputs
                context2, _, _= encoder2(lang1, lengths1)
                decoder_hidden = decoder.init_hidden(batch_size)
            elif encoder.model_type == 'rnn':
                if (encoder.bidirectional) | (encoder.num_layers == 2):
                    context = torch.cat((encoder_hidden[-2], encoder_hidden[-1]), dim=1)
                else:
                    context = encoder_hidden[-1]
                context = context.unsqueeze(0)
                decoder_hidden = (encoder_hidden, encoder_cell)

            if dmodel_type == 'attn' and emodel_type != 'cnn':
                context = encoder_outputs
                encoder_hidden = translator.combine_directions(encoder_hidden)
                encoder_cell = translator.combine_directions(encoder_cell)
                #print('eh:', encoder_hidden.shape)
                decoder_hidden = (encoder_hidden, encoder_cell)

            # make this 1 x batchsize
            decoder_input = torch.tensor([translator.Language.SOS_IDX] * batch_size, device=device)

            decoder_full_out = torch.zeros(batch_size, target_length, output_vocab.n_words, device=device)

            for di in range(target_length):
                decoder_input = decoder_input.unsqueeze(0)
                # print("SL:", decoder_input.shape)
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, context, context2)
                topv, topi = decoder_output.topk(1)
                decoder_full_out[:, di] = decoder_output
                decoder_input = topi.squeeze().detach()  # detach from history as input

            decoder_full_out = decoder_full_out.transpose(1, 2)
            # print(decoder_full_out.shape, lang2.shape)
            loss = criterion(decoder_full_out, lang2)
            # print(loss)
            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)
    return val_loss


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
    ap.add_argument("-gc", "--grad_clip", type=float, default=0.1,
                    help="")
    ap.add_argument("-mo", "--momentum", type=float, default=0.99,
                    help="")
    ap.add_argument("-op", "--optimizer", default='sgd',
                    help="sgd or adam")
    ap.add_argument("-un", "--use_nesterov", action="store_true",
                    help="")

    ap.add_argument("-db", "--debug", action="store_true",
                    help="Print some extra info")
    ap.add_argument("-ls", "--lr_schedule", type=int, default=5,
                    help="use a min change scheduler with this pateince, 0 means don't use")

    ap.add_argument("-le", "--load_encoder",
                    help="encoder to load")

    ap.add_argument("-ld", "--load_decoder",
                    help="encoder to load")

    ap.add_argument("-tt", "--test", action="store_true",
                    help="Print some extra info")

    args = vars(ap.parse_args())

    #define our model and training run paramaters
    print_every = args['print_iter']
    lang_dir = args['source_dir']
    lang1, lang2 = args['target_pair'].split('-')

    epochs = args['epochs']
    batch_size = args['batch_size']

    max_length = args['max_length']
    max_vocab = args['max_vocab']

    dmodel_type = args['decoder_model_type']
    emodel_type = args['encoder_model_type']
    num_layers = args['num_layers']
    bidirectional = not args['single_direction']
    hidden_size = args['hidden_size']
    embed_dim = args['embed_dim']

    teacher_forcing_ratio = args['teacher_force']

    optimizer = args['optimizer']
    learning_rate = args['learning_rate']
    momentum = args['momentum']
    use_nesterov = args['use_nesterov']
    grad_clip = args['grad_clip']
    lr_schedule = args['lr_schedule']

    encoder_path = args['load_encoder']
    decoder_path = args['load_decoder']

    TEST = args['test']
    DEBUG = args['debug']

    #TODO: confirm our models pair correctly
    #rnn/rnn - can be any layers, or direction
    #cnn/rnn - can't be bidirectional or more than one later
    #rnn/attn - rnn must be bidirectional, attn not, can be any layers
    #other pairings not supported.

    #hack, use extension of first file as language type
    lang_label = lang1

    #need to run these both to avoid passing device around everywhere
    translator.init_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loads the data from the two files and creates two vocabs and a set of string sentence tuples
    input_vocab, train_input_sentences, val_input_sentences, test_input_sentences = translator.loadData(lang_dir, lang1, max_vocab)
    output_vocab, train_output_sentences , val_output_sentences , test_output_sentences = translator.loadData(lang_dir, lang2, max_vocab)

    translator.cleanSentences(train_input_sentences,train_output_sentences )
    translator.cleanSentences(val_input_sentences, val_output_sentences)
    translator.cleanSentences(test_input_sentences, test_output_sentences)

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

        print(input_vocab.index2word[0:15])
        print(output_vocab.index2word[0:15])

    # define our encoder and decoder
    encoder = None
    decoder = None

    if emodel_type == 'rnn':
        encoder = translator.EncoderRNN(input_vocab.n_words, embed_dim, num_layers, bidirectional).to(device)
    elif emodel_type == 'cnn':
        encoder = translator.EncoderCNN(input_vocab.n_words, embed_dim, max_length).to(device)
        encoder2 = translator.EncoderCNN(input_vocab.n_words, embed_dim, max_length).to(device)
    else:
        print("unknown model_type", emodel_type)
        exit(1)

    decoder = None

    if dmodel_type == 'rnn':
        decoder =translator.DecoderRNN(hidden_size, output_vocab.n_words, num_layers).to(device)
    elif dmodel_type == 'attn':
        decoder = translator.BahdanauAttnDecoderRNN(hidden_size, output_vocab.n_words, num_layers, dropout_p=0.1).to(device)
    else: #TODO: implement other model types
        print("unknown model_type", dmodel_type)
        exit(1)

    if encoder_path is not None:
        translator.load_model(encoder, encoder_path)

    if decoder_path is not None:
        translator.load_model(decoder, decoder_path)

    save_prefix = os.path.join(args['model_directory'], lang_label, emodel_type + '-' + dmodel_type)
    os.makedirs(save_prefix, exist_ok=True)
    print("Max Vocab Size:", max_vocab, ", Max Sentence Length", max_length)
    print("Running on ", device)
    print("Using model types:", emodel_type, '/', dmodel_type)
    print("saving in:", save_prefix, flush=True)
    print("Layers:", num_layers, "bidirectionl" if bidirectional else "unidirectional")
    print("Hidden Size:", hidden_size, ", Embd Dim:", embed_dim)
    print("Batch Size:", batch_size, "patience", lr_schedule)
    print("Teacher Force:", teacher_forcing_ratio)

    if optimizer == 'sgd':
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, nesterov = use_nesterov, momentum=momentum)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate, nesterov = use_nesterov, momentum=momentum)
    elif optimizer == 'adam':
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    else:
        print("Must specify sgd or adam")
        exit(1)

    if (lr_schedule):
        encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min', verbose = True, patience = lr_schedule)
        decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'min', verbose = True, patience = lr_schedule)

    criterion = nn.NLLLoss(ignore_index=translator.Language.PAD_IDX)

    #this is here so we can test large changes without having to wait for a full training epcoch
    if DEBUG:
        print("Testing eval_model...")
        vl = eval_model(encoder, decoder, val_loader, criterion)
        print("val loss:", vl)

        print("Testing BLEU...")
        blu = translator.evaluateBLUE(val_input_index, val_output_index, output_vocab, encoder, decoder, max_length)
        print("Val Blue:", blu)

    #this is here so we can test large changes without having to wait for a full training epcoch
    if TEST:
        test_input_index = translator.indexSentences(input_vocab, test_input_sentences)
        test_output_index = translator.indexSentences(output_vocab, test_output_sentences)

        test_dataset = translator.PairsDataset(test_input_index, test_output_index, max_length)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                 collate_fn=test_dataset.vocab_collate_func)

        print("Evaluating TEST set...")
        vl = eval_model(encoder, decoder, test_loader, criterion)
        print("TEST loss:", vl)

        print("Testing BLEU...")
        blu = translator.evaluateBLUE(test_input_index, test_output_index, output_vocab, encoder, decoder, max_length)
        print("TEST Blue:", blu)



    print("Begin Training!", flush=True)

    best_bleu = 0
    start = time.time()
    for epoch in range(epochs):

        print_loss_total = 0  # Reset every print_every

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
            context2 = None

            if emodel_type == 'cnn':
                context = encoder_outputs
                context2, _, _ = encoder2(lang1, lengths1)
                decoder_hidden = decoder.init_hidden(batch_size)
            elif emodel_type == 'rnn' and dmodel_type != 'attn':
                if (bidirectional) | (num_layers == 2):
                    context = torch.cat((encoder_hidden[-2], encoder_hidden[-1]), dim=1)
                else:
                    context = encoder_hidden[-1]
                context = context.unsqueeze(0)

                decoder_hidden = (encoder_hidden, encoder_cell)

            if dmodel_type == 'attn' and emodel_type != 'cnn':
                context = encoder_outputs
                encoder_hidden = translator.combine_directions(encoder_hidden)
                encoder_cell = translator.combine_directions(encoder_cell)
                # print('eh:', encoder_hidden.shape)
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
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, context, context2)
                    decoder_full_out[:, di] = decoder_output

                    decoder_input = target_tensor[di]  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_input = decoder_input.unsqueeze(0)
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, context, context2)
                    topv, topi = decoder_output.topk(1)
                    decoder_full_out[:, di] = decoder_output
                    decoder_input = topi.squeeze().detach()  # detach from history as input

            decoder_full_out = decoder_full_out.transpose(1, 2)

            loss = criterion(decoder_full_out, lang2)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            encoder_optimizer.step()
            decoder_optimizer.step()

            loss = loss.item()

            print_loss_total += loss

            if (print_every > -1) & (i % print_every == 0) & (i > 0):
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('epoch %d : %s (%d %d%%) Loss: %.4f' %  (epoch, translator.timeSince(start, i / total_batches),
                                             i, i / total_batches * 100, print_loss_avg), flush=True)

        #print out the final set at the end of the epoch
        if (print_every > -1) & (i > 0):
            print_loss_avg = print_loss_total / (total_batches % print_every)
            print_loss_total = 0
            print('epoch %d : %s (%d %d%%) Loss: %.4f' % (epoch, translator.timeSince(start, i / total_batches),
                                                          i, i / total_batches * 100, print_loss_avg), flush=True)


        print("Computing VAL", flush=True)

        val_loss = eval_model(encoder, decoder, val_loader, criterion)
        print("Val loss:", val_loss, flush=True)

        if lr_schedule:
            encoder_scheduler.step(val_loss)
            decoder_scheduler.step(val_loss)

        blu = translator.evaluateBLUE(val_input_index, val_output_index, output_vocab, encoder, decoder, max_length)
        print("Val Blue:", blu, flush=True)

        if blu > best_bleu:
            best_bleu = blu
            print("Saving model.", flush=True)
            translator.save_model(encoder, decoder, save_prefix, epoch)


