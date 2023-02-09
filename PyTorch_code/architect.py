import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152, ResNet152_Weights
from collections import namedtuple
from structures import FibonacciHeap
from labml_nn.sampling import Sampler
from labml_nn.sampling.temperature import TemperatureSampler

import numpy as np
import torch

BeamMemory = namedtuple('Cand',
                        ['lstm', 'log_prob_seq', 'last_word_id', 'word_id_seq'])

BeamMemory2 = namedtuple('Cand',
                         ['embedding', 'states', 'log_prob_seq', 'last_word_id', 'word_id_seq'])


class Encoder(nn.Module):
    def __init__(self,
                 embed_size: int):
        super(Encoder, self).__init__()
        resnet = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)

        for p in resnet.parameters():
            p.requires_grad_(False)

        mods = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*mods)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 lstm_dim: int,
                 voc_size: int,
                 start_word_id: int,
                 unk_word_id: int,
                 end_word_id: int,
                 ind2word: dict,
                 prob_mass: float = 0.1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTMCell(embed_dim, lstm_dim)
        self.lstm2 = nn.LSTM(input_size=embed_dim,
                             hidden_size=lstm_dim,
                             num_layers=1,
                             batch_first=True,
                             dropout=0.4)
        self.lstm_dim = lstm_dim
        self.emb_layer = nn.Embedding(voc_size, embed_dim)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(lstm_dim, voc_size)
        self.voc_size = voc_size
        self.start_word_id = start_word_id
        self.end_word_id = end_word_id
        self.layer_norm = nn.LayerNorm(lstm_dim)
        self.nucleus = NucleusSampler(prob_mass, TemperatureSampler(1.))
        self.unk_word_id = unk_word_id
        self.lstm_dim = lstm_dim
        self.ind2word = ind2word

    def forward(self, features, caption, mode='nic_paper'):
        if mode == 'nic_paper':
            return self.nic_paper(features, caption)
        else:
            return self.shwetank(features, caption)

    def shwetank(self, features, caption):
        """
        Original code from Panwar, Shwetank. NIC-2015-Pytorch. 2019. Github repository.
        https://github.com/pshwetank/NIC-2015-Pytorch
        This is being used to test how well Shwetank's sampler is compared to beamsearch and nucleus sampling
        """

        batch_size = features.size(0)
        self.hidden = ((torch.zeros((1,
                                     batch_size,
                                     self.lstm_dim))).to(self.device),
                       (torch.zeros((1,
                                     batch_size,
                                     self.lstm_dim))).to(self.device))
        x = self.emb_layer(caption[:, :-1])
        y = torch.cat((features.unsqueeze(1), x), dim=1)
        y, self.hidden = self.lstm2(y, self.hidden)
        return self.linear(y)

    def nic_paper(self, features, caption):
        x = self.lstm(features)
        outputs = list()

        for ind in range(caption.size(1) - 1):
            word = caption[:, ind].clone()
            y = self.emb_layer(word)
            x = self.lstm(y, x)
            # y = self.dropout(x[0])
            y = self.layer_norm(x[0])
            y = self.linear(y)
            outputs.append(F.log_softmax(y, dim=1))

        return torch.stack(outputs, dim=1)

    def nic_beam_search(self,
                        features,
                        beam_size: int = 3,
                        max_seq_len: int = 12):
        self.eval()
        features = features.view(1, -1)
        x = self.lstm(features)
        top_k_cand = FibonacciHeap()
        top_k_cand.insert(0., BeamMemory(x, [], self.start_word_id, []))

        for msl in range(max_seq_len):
            tmp_top_k_cand = FibonacciHeap()
            stop = True

            while top_k_cand.n > 0:
                node = top_k_cand.extract_max()
                log_prob_sum = node.key
                lstm, log_prob_seq, last_word_id, word_id_seq = node.val

                if msl > 0 and last_word_id == self.end_word_id:
                    tmp_top_k_cand.insert(node.key, node.val)
                else:
                    stop = False
                    word = features.new_tensor([last_word_id], dtype=torch.long)
                    y = self.emb_layer(word)
                    lstm = self.lstm(y, lstm)
                    # y = self.dropout(lstm[0])
                    y = self.layer_norm(lstm[0])
                    y = self.linear(y)
                    y = F.log_softmax(y, dim=1).squeeze(0)
                    y[self.start_word_id] += float('-inf')
                    y[self.unk_word_id] += float('-inf')
                    y[self.end_word_id] += float('-inf')
                    y, ind = torch.sort(y, descending=True)

                    for k in range(beam_size):
                        log_prob, word_id = y[k], ind[k]
                        log_prob = float(log_prob)
                        word_id = int(word_id)
                        tmp_top_k_cand.insert(log_prob_sum + log_prob,
                                              BeamMemory(lstm,
                                                         log_prob_seq + [log_prob],
                                                         word_id,
                                                         word_id_seq + [word_id]))

            for k in range(beam_size):
                if tmp_top_k_cand.n > 0:
                    node = tmp_top_k_cand.extract_max()
                    top_k_cand.insert(node.key, node.val)
                else:
                    break

            if stop:
                break

        captions = list()
        log_prob_sums = list()

        while top_k_cand.n > 0:
            node = top_k_cand.extract_max()
            words = list()

            for ind in node.val.word_id_seq:
                words.append(self.ind2word[ind])

            captions.append(' '.join(words))
            log_prob_sums.append(node.key)

        return captions, log_prob_sums

    def shwetank_beam_search(self,
                             features,
                             beam_size: int = 3,
                             max_seq_len: int = 12):
        self.eval()
        states = (torch.randn(1, 1, self.lstm_dim).to(features.device), torch.randn(1, 1,
                                                                                    self.lstm_dim).to(
            features.device))
        out, states = self.lstm2(features, states)
        word_id = np.argmax(self.linear(out).cpu().detach().numpy().flatten())
        word = torch.cuda.LongTensor([word_id])
        # word = features.new_tensor([word_id], dtype=torch.long)
        embedding = self.emb_layer(word)
        embedding = embedding.view(1, embedding.size(0), -1)
        top_k_cand = FibonacciHeap()
        top_k_cand.insert(0., BeamMemory2(embedding, states, [], self.start_word_id, []))

        for msl in range(max_seq_len):
            tmp_top_k_cand = FibonacciHeap()
            stop = True

            while top_k_cand.n > 0:
                node = top_k_cand.extract_max()
                prob_sum = node.key
                embedding, states, log_prob_seq, last_word_id, word_id_seq = node.val

                if msl > 0 and last_word_id == self.end_word_id:
                    tmp_top_k_cand.insert(node.key, node.val)
                else:
                    stop = False
                    output, states = self.lstm2(embedding, states)
                    y = self.linear(output).squeeze()
                    y[self.start_word_id] += float('-inf')
                    y[self.unk_word_id] += float('-inf')
                    # y[self.end_word_id] += float('-inf')
                    y, ind = torch.sort(y, descending=True)

                    for k in range(beam_size):
                        prob, word_id = y[k], ind[k]
                        prob = float(prob)
                        word_id = int(word_id)
                        # word = features.new_tensor([word_id], dtype=torch.long)
                        word = torch.cuda.LongTensor([word_id])
                        embedding = self.emb_layer(word)
                        embedding = embedding.view(1, embedding.size(0), -1)
                        tmp_top_k_cand.insert(prob_sum + prob,
                                              BeamMemory2(embedding,
                                                          states,
                                                          log_prob_seq + [prob],
                                                          word_id,
                                                          word_id_seq + [word_id]))

            for k in range(beam_size):
                if tmp_top_k_cand.n > 0:
                    node = tmp_top_k_cand.extract_max()
                    top_k_cand.insert(node.key, node.val)
                else:
                    break

            if stop:
                break

        captions = list()
        log_prob_sums = list()

        while top_k_cand.n > 0:
            node = top_k_cand.extract_max()
            words = list()

            for ind in node.val.word_id_seq:
                if ind != self.end_word_id:
                    words.append(self.ind2word[ind])

            captions.append(' '.join(words))
            log_prob_sums.append(node.key)

        return captions, log_prob_sums

    def nic_sample(self,
                   features,
                   max_len=20):
        self.eval()
        samples = []
        features = features.view(1, -1)
        lstm = self.lstm(features)
        last_word_id = features.new_tensor([self.start_word_id], dtype=torch.long)

        for i in range(max_len):
            y = self.emb_layer(last_word_id)
            lstm = self.lstm(y, lstm)
            y = self.layer_norm(lstm[0])
            y = self.linear(y)
            y = F.log_softmax(y, dim=1).squeeze(0)
            y[self.start_word_id] += float('-inf')
            y[self.unk_word_id] += float('-inf')
            word_id = np.argmax(y.cpu().detach().numpy().flatten())
            last_word_id = features.new_tensor([word_id], dtype=torch.long)

            if int(last_word_id) == self.end_word_id:
                break

            samples.append(int(last_word_id))

        return samples

    def shwetank_sample(self, inputs, states=None, max_len=20):
        """
        Original code from Panwar, Shwetank. NIC-2015-Pytorch. 2019. Github repository.
        https://github.com/pshwetank/NIC-2015-Pytorch
        This is being used to test how well Shwetank's sampler is compared to beamsearch and nucleus sampling
        """
        self.eval()
        samples = []

        if states is None:
            states = (torch.randn(1, 1, self.lstm_dim).to(inputs.device), torch.randn(1, 1,
                                                                                      self.lstm_dim).to(
                inputs.device))
        out, states = self.lstm2(inputs, states)

        linear1 = self.linear(out)
        idx = np.argmax(linear1.cpu().detach().numpy().flatten())
        start_capt = torch.cuda.LongTensor([idx])
        embedding = self.emb_layer(start_capt)
        embedding = embedding.view(1, embedding.size(0), -1)
        for i in range(max_len):
            output, states = self.lstm2(embedding, states)

            linear = self.linear(output)
            idx = np.argmax(linear.cpu().detach().numpy().flatten())
            if int(idx) == self.end_word_id:
                break
            samples.append(int(idx))
            start_capt = torch.cuda.LongTensor([idx])
            embedding = self.emb_layer(start_capt)
            embedding = embedding.view(1, embedding.size(0), -1)

        return samples

    def nic_nucleus(self, features, max_len=20):
        self.eval()
        samples = []
        features = features.view(1, -1)
        lstm = self.lstm(features)
        last_word_id = features.new_tensor([self.start_word_id], dtype=torch.long)

        for i in range(max_len):
            y = self.emb_layer(last_word_id)
            lstm = self.lstm(y, lstm)
            y = self.layer_norm(lstm[0])
            y = self.linear(y)
            y = F.log_softmax(y, dim=1)
            y[:, self.start_word_id] += float('-inf')
            y[:, self.unk_word_id] += float('-inf')
            last_word_id = self.nucleus(y)

            if int(last_word_id) == self.end_word_id:
                break

            samples.append(int(last_word_id))

        return samples

    def shwetank_nucleus(self, features, max_len=20):
        self.eval()
        states = (torch.randn(1, 1, self.lstm_dim).to(features.device), torch.randn(1, 1,
                                                                                  self.lstm_dim).to(
            features.device))
        out, states = self.lstm2(features, states)
        y = self.linear(out)
        word_ind = self.nucleus(y)
        embedding = self.emb_layer(word_ind)
        embedding = embedding.view(1, embedding.size(0), -1)
        samples = []

        for i in range(max_len):
            out, states = self.lstm2(embedding, states)
            y = self.linear(out)
            word_ind = self.nucleus(y)

            if int(word_ind) == self.end_word_id:
                break

            samples.append(int(word_ind))
            embedding = self.emb_layer(word_ind)
            embedding = embedding.view(1, embedding.size(0), -1)

        return samples


class NucleusSampler(Sampler):
    """
    Derived from https://nn.labml.ai/sampling/nucleus.html
    Need to add citations
    """

    def __init__(self, p: float, sampler: Sampler):
        self.p = p
        self.sampler = sampler
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, logits: torch.Tensor):
        probs = self.softmax(logits)
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus = cum_sum_probs < self.p
        nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
        sorted_log_probs = torch.log(sorted_probs)
        sorted_log_probs[~nucleus] = float('-inf')
        sampled_sorted_indexes = self.sampler(sorted_log_probs)
        res = indices.gather(-1, sampled_sorted_indexes.unsqueeze(-1))
        return res.squeeze(-1)
