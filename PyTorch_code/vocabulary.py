from collections import Counter
from torchtext.data.utils import get_tokenizer
from os.path import exists, basename, splitext, join, dirname

import json
import pickle

# spacy_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
spacy_tokenizer = get_tokenizer("basic_english")
script_dir = dirname(__file__)


class Vocabulary(object):
    """
    Adapted from: Shwetank Panwar "NIC-2015-Pytorch" (May 1, 2020)
    https://github.com/pshwetank/NIC-2015-Pytorch
    I have rewritten and improved performance from Panwar's vocabulary class by
    leveraging divide and conquer algorithm, torchtext's get_tokenizer, and removed
    unnecessary items.
    """

    def __init__(self,
                 annotation_fp: str,
                 vocab_ext: str = '.voc',
                 vocab_threshold: int = -1,
                 reload: bool = False,
                 start_word: str = '<start>',
                 end_word: str = '<end>',
                 unknown_word: str = '<unknown>'):
        self.annotation_fp = annotation_fp
        self.vocab_fp = join(script_dir,
                             '{0}{1}_{2}thresh'.format(splitext(basename(self.annotation_fp))[0],
                                                       vocab_ext,
                                                       vocab_threshold))
        self.vocab_threshold = vocab_threshold
        self.start_word = start_word
        self.end_word = end_word
        self.unknown_word = unknown_word
        self.__word2ind = dict()
        self.__ind2word = dict()
        self.__ind = 0
        self.__captions = dict()
        self.__load_vocab(reload)

    @property
    def word2ind(self) -> dict:
        return self.__word2ind

    @property
    def ind2word(self) -> dict:
        return self.__ind2word

    @property
    def ind(self) -> int:
        return self.__ind

    @property
    def captions(self) -> dict:
        return self.__captions

    def add_word(self, word: str):
        if word not in self.__word2ind:
            self.__word2ind[word] = self.__ind
            self.__ind2word[self.__ind] = word
            self.__ind += 1

    def __load_vocab(self, reload):
        if not reload and exists(self.vocab_fp):
            with open(self.vocab_fp, 'rb') as f:
                vocab_class = pickle.load(f)
                self.__word2ind = vocab_class.word2ind
                self.__ind2word = vocab_class.ind2word
                self.__ind = vocab_class.ind
                self.__captions = vocab_class.captions
        else:
            self.__word2ind = dict()
            self.__ind2word = dict()
            self.__ind = 0
            self.add_word(self.unknown_word)
            self.add_word(self.start_word)
            self.add_word(self.end_word)
            self.__generate_vocab()

            with open(self.vocab_fp, 'wb') as f:
                pickle.dump(self, f)

    def __generate_vocab(self):
        vocab_count = Counter()

        def proc_caption(row: dict):
            tokens = spacy_tokenizer(row['caption'].lower())
            self.__captions[row['id']] = tokens
            vocab_count.update(tokens)

        def div_alg(left, right, func):
            if len(left) > 1:
                lm = len(left) // 2
                div_alg(left[:lm], left[lm:], func)
            else:
                func(left[0])

            if len(right) > 1:
                rm = len(right) // 2
                div_alg(right[:rm], right[rm:], func)
            else:
                func(right[0])

        with open(self.annotation_fp, 'r') as f:
            data = json.load(f)

        assert type(data) == dict,\
            'annotation file format "{}" is not supported'.format(type(data))

        if 'annotations' in data:
            annotations = data['annotations']
            m = len(annotations) // 2
            div_alg(annotations[:m], annotations[m:], proc_caption)
            words = [word for word, count in vocab_count.items() if count >= self.vocab_threshold]
            m = len(words) // 2
            div_alg(words[:m], words[m:], self.add_word)

    def __call__(self, token, word2ind=True):
        if word2ind:
            if token not in self.__word2ind:
                return self.__word2ind[self.unknown_word]

            return self.__word2ind[token]
        else:
            if token not in self.__ind2word:
                raise ValueError('Index {} is not found in ind2word vector'.format(token))

            return self.__ind2word[token]

    def __len__(self) -> int:
        return len(self.__word2ind)
