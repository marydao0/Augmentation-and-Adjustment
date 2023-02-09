from test_dataloader import CoCoLoader
from transforms import text_trans, aug_img_trans, aug_img_trans2, img_trans
from architect import Encoder, Decoder
from numpy import exp as np_exp
from sys import stdout
from os.path import join, dirname, exists
from os import makedirs
from time import time

import torch

# Requires image and annotations from https://cocodataset.org/#download

train_img_dir = r'\cocoapi\images\train2017'
train_ann_fp = r'\cocoapi\annotations\captions_train2017.json'
test_img_dir = r'\cocoapi\images\test2017'
test_ann_fp = r'\cocoapi\annotations\image_info_test2017.json'

file_dir = dirname(__file__)
model_dir = join(file_dir, 'models')


if __name__ == '__main__':
    embed_size = 300
    lstm_dim = 128
    num_of_epochs = 3
    batch_size = 128
    vocab_thresh = 8
    train_name = 'nic_paper_aug'
    log_file = '%s_training_loss.txt' % train_name

    log_f = open(log_file, 'w')

    if not exists(model_dir):
        makedirs(model_dir)

    train_loader = CoCoLoader(train_img_dir,
                              train_ann_fp,
                              batch_size=batch_size,
                              img_transform=aug_img_trans,
                              target_transform=text_trans,
                              vocab_threshold=vocab_thresh,
                              load_captions=True)

    vocab = train_loader.vocab
    vocab_size = len(vocab)

    encoder = Encoder(embed_size)
    decoder = Decoder(embed_size,
                      lstm_dim,
                      vocab_size,
                      vocab.word2ind[vocab.start_word],
                      vocab.word2ind[vocab.unknown_word],
                      vocab.word2ind[vocab.end_word],
                      vocab.ind2word)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)
    loss_fn = torch.nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else torch.nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters())
    num_of_iter = int(train_loader.num_of_iter)

    optimizer = torch.optim.Adam(
        params,
        lr=0.001
    )
    epoch_start_time = time()

    for epoch in range(1, num_of_epochs + 1):
        if epoch == 1:
            epoch_print = f'---------------------------------------------Epoch {epoch}/{num_of_epochs}---------------------------------------------'
        else:
            epoch_print = f'---------------------------------------------Epoch {epoch}/{num_of_epochs}, Prev Epoch Time: {(time() - epoch_start_time)}---------------------------------------------'

        print('\r' + epoch_print, end='')
        stdout.flush()
        log_f.write(epoch_print + '\n')
        log_f.flush()
        step = 1
        epoch_start_time = time()
        load_start_time = time()

        for img_batch, captions in train_loader:
            step_start_time = time()
            img_batch = img_batch.to(device)
            captions = captions.to(device)
            decoder.zero_grad()
            encoder.zero_grad()
            enc_feats = encoder(img_batch)
            probs = decoder(enc_feats, captions) # , mode='shwetank'

            if probs.shape[1] < captions.shape[1]:
                captions = captions[:, 1:].contiguous().view(-1)
            else:
                captions = captions.contiguous().view(-1)

            probs = probs.view(-1, vocab_size)
            loss = loss_fn(probs, captions)
            loss.backward()
            optimizer.step()
            update = '\n \t \t \t step {:}/{:} \t loss {:.3f} \t perplexity {:.3f} \t step time: {:}, \t load time: {:}'.format(
                step,
                num_of_iter,
                loss,
                np_exp(loss.item()),
                (time() - step_start_time),
                (step_start_time - load_start_time)
            )
            print('\r' + update, end='')
            stdout.flush()
            log_f.write(update + '\n')
            log_f.flush()
            step += 1
            load_start_time = time()

        torch.save(decoder.state_dict(),
                   join(model_dir, '{}_decoder-{}.pkl'.format(train_name, epoch)))
        torch.save(encoder.state_dict(),
                   join(model_dir, '{}_encoder-{}.pkl'.format(train_name, epoch)))

    log_f.close()
