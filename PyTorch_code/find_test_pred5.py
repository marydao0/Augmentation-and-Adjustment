from test_dataloader import CoCoLoader
from transforms import text_trans, aug_img_trans, aug_img_trans2, img_trans
from architect import Encoder, Decoder
from os.path import join, dirname, exists
from os import makedirs
from predict import get_predict

import torch

# Requires image and annotations from https://cocodataset.org/#download

test_img_dir = r'/Users/marydao/Downloads/NIC-2015-Pytorch-master/cocoapi/images/val2017'
train_ann_fp = r'/Users/marydao/Downloads/NIC-2015-Pytorch-master/cocoapi/annotations/captions_train2017.json'
test_ann_fp = r'/Users/marydao/Downloads/NIC-2015-Pytorch-master/cocoapi/annotations/captions_val2017.json'

file_dir = dirname(__file__)
model_dir = join(join(file_dir, 'models'), '04_train4_py_models')

if __name__ == '__main__':
    embed_size = 300
    lstm_dim = 128
    batch_size = 1
    vocab_thresh = 8
    prob_mass = 0.13

    if not exists(model_dir):
        makedirs(model_dir)

    test_loader = CoCoLoader(test_img_dir,
                             test_ann_fp,
                             vocab_fp=train_ann_fp,
                             batch_size=batch_size,
                             img_transform=img_trans,
                             target_transform=text_trans,
                             vocab_threshold=vocab_thresh,
                             load_captions=True,
                             image_to_ann=True)

    vocab = test_loader.vocab
    vocab_size = len(vocab)
    encoder = Encoder(embed_size)
    encoder.eval()

    decoder = Decoder(embed_size,
                      lstm_dim,
                      vocab_size,
                      vocab.word2ind[vocab.start_word],
                      vocab.word2ind[vocab.unknown_word],
                      vocab.word2ind[vocab.end_word],
                      vocab.ind2word,
                      prob_mass=prob_mass)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder.eval()
    decoder.load_state_dict(torch.load(join(model_dir, 'decoder-1.pkl'), map_location=torch.device(device)))
    encoder.load_state_dict(torch.load(join(model_dir, 'encoder-1.pkl'), map_location=torch.device(device)),
                            strict=False)

    encoder.to(device)
    decoder.to(device)
    get_predict(test_loader=test_loader,
                encoder=encoder,
                decoder=decoder,
                device=device,
                vocab=vocab,
                mode='nucleus',
                beam_size=20,
                max_seq_len=13)
