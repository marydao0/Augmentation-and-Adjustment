from numpy import squeeze as np_squeeze
import matplotlib.pyplot as plt
from textwrap import wrap

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize


def get_predict(test_loader,
                encoder,
                decoder,
                device,
                vocab,
                shwetank=False,
                mode='sample',
                beam_size: int = 3,
                max_seq_len: int = 12):
    if mode not in ('sample', 'beam', 'nucleus'):
        raise ValueError('mode must be sample, beam, or nucleus')

    # for data in test_loader:
    data = next(iter(test_loader))
    orig_image, image, true_captions = data[0], data[1], data[2]
    features = encoder(image.to(device)).unsqueeze(1)

    true_captions = [cap[0] for cap in true_captions]

    if mode == 'beam':
        if shwetank:
            output = decoder.shwetank_beam_search(features, beam_size, max_seq_len)
        else:
            output = decoder.nic_beam_search(features, beam_size, max_seq_len)

        print('beam top %s:' % beam_size, output)
        caption = output[0][0]
    elif mode == 'nucleus':
        if shwetank:
            output = decoder.shwetank_nucleus(features, max_seq_len)
            beam_output = decoder.shwetank_beam_search(features, beam_size, max_seq_len)
            sample_output = decoder.shwetank_sample(features, max_len=max_seq_len)
        else:
            output = decoder.nic_nucleus(features, max_seq_len)
            beam_output = decoder.nic_beam_search(features, beam_size, max_seq_len)
            sample_output = decoder.nic_sample(features, max_len=max_seq_len)

        caption = ' '.join([vocab.ind2word[i] for i in output])

        print('true captions:', true_captions)
        print('sample:', ' '.join([vocab.ind2word[i] for i in sample_output]))
        print('beam:', beam_output[0][0])
        print('nucleus:', caption)

        caption = [vocab.ind2word[i] for i in output]
        sample_caption = [vocab.ind2word[i] for i in sample_output]
        beam_caption = beam_output[0][0].split()

        reference = [word_tokenize(cap.lower()) for cap in true_captions]
        chencherry = SmoothingFunction()
        print('sample:', ' '.join(sample_caption))
        print('\tBLEU-1: %f' % sentence_bleu(reference, sample_caption, weights=(1, 0, 0, 0),
                                             smoothing_function=chencherry.method2))
        print('\tBLEU-4: %f' % sentence_bleu(reference, sample_caption, weights=(0.25, 0.25, 0.25, 0.25),
                                             smoothing_function=chencherry.method2), '\n')

        print('beam:', ' '.join(beam_caption))
        print('\tBLEU-1: %f' % sentence_bleu(reference, beam_caption, weights=(1, 0, 0, 0),
                                             smoothing_function=chencherry.method2))
        print('\tBLEU-4: %f' % sentence_bleu(reference, beam_caption, weights=(0.25, 0.25, 0.25, 0.25),
                                             smoothing_function=chencherry.method2), '\n')

        print('nucleus:', ' '.join(caption))
        print('\tBLEU-1: %f' % sentence_bleu(reference, caption, weights=(1, 0, 0, 0),
                                             smoothing_function=chencherry.method2))
        print('\tBLEU-4: %f' % sentence_bleu(reference, caption, weights=(0.25, 0.25, 0.25, 0.25),
                                             smoothing_function=chencherry.method2), '\n')
        print(caption)

    else:
        if shwetank:
            output = decoder.shwetank_sample(features, max_len=max_seq_len)
        else:
            output = decoder.nic_sample(features, max_len=max_seq_len)

    caption = ' '.join([vocab.ind2word[i] for i in output])

    caption = "\n".join(wrap(' '.join(caption), 60))
    plt.imshow(np_squeeze(orig_image))
    plt.title(caption)
    plt.show()

