
import spacy
from collections import Counter
import torch

class Vocabulary(object):
    def __init__(self, freq_threshold, spacy_eng=None):
        self.start_word = "<SOS>"
        self.end_word = "<EOS>"
        self.pad_word = "<PAD>"
        self.unk_word = "<UNK>"

        self.itos = {0: self.pad_word, 1: self.start_word, 2: self.end_word, 3: self.unk_word}
        self.stoi = {self.pad_word: 0, self.start_word: 1, self.end_word: 2, self.unk_word: 3}
        
        self.freq_threshold = freq_threshold
        if spacy_eng is None:
            self.spacy_eng = spacy.load('en_core_web_sm')
        else:
            self.spacy_eng = spacy_eng

    def __len__(self):
        return len(self.itos)

    def tokenizer_eng(self, text):
        tokenizer = [tok.text.lower() for tok in self.spacy_eng.tokenizer(text)]
        return tokenizer
        
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class CaptionCollate:
    def __init__(self, pad_idx, batch_first=True):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        batch.sort(key=lambda x: len(x[1]), reverse=True)
        (images, captions) = zip(*batch)
        
        imgs = [img.unsqueeze(0) for img in images]
        imgs = torch.cat(imgs, dim=0)
        
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for idx, cap in enumerate(captions):
            end = lengths[idx]
            targets[idx, :end] = cap[:end]
        return imgs, targets, lengths
