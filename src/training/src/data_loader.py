import json
import numpy as np
from collections import Counter

from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from . import utils

import torch


def get_sample_weights(labels):
    counter = Counter(labels)
    counter = dict(counter)
    for k in counter:
        counter[k] = 1 / counter[k]
    sample_weights = np.array([counter[l] for l in labels])
    return sample_weights


def load_data(args):
    # chunk your dataframes in small portions
    chunks = pd.read_csv(args.data_path,
                         usecols=[args.text_column, args.label_column],
                         chunksize=args.chunksize,
                         encoding=args.encoding,
                         nrows=args.max_rows,
                         sep=args.sep)
    texts = []
    labels = []
    for df_chunk in tqdm(chunks):
        aux_df = df_chunk.copy()
        aux_df = aux_df[~aux_df[args.text_column].isnull()]
        aux_df = aux_df[(aux_df[args.text_column].map(len) > 3)]
                        # & (aux_df[args.text_column].map(len) < 600)]
        aux_df['processed_text'] = (aux_df[args.text_column]
                                    .map(lambda text: utils.process_text(args.steps, text)))
        texts += aux_df['processed_text'].tolist()
        labels += aux_df[args.label_column].tolist()

    if args.group_labels == 'binarize':
        '''
        This part is adapted to the data I was dealing with
        where I basically convert 5-star ratings to binary 
        by discarding the number 3 and 4 and keeping 1, 2 and 5

        1, 2 become the negative label
        5 becomes the positive one

        '''

        clean_data = [(text, label) for (text, label) in zip(
            texts, labels) if label not in [3]]

        texts = [text for (text, label) in clean_data]
        labels = [label for (text, label) in clean_data]

        labels = list(
            map(lambda l: {1: 0, 2: 0, 4:1, 5: 1}[l], labels))

        count_minority = labels.count(0)

        if bool(args.balance):

            balanced_texts = ([text for text, label in zip(texts, labels) if label == 0] +
                              [text for text, label in zip(texts, labels) if label == 1][:int(args.ratio * count_minority)])
            balanced_labels = ([label for text, label in zip(texts, labels) if label == 0] +
                               [label for text, label in zip(texts, labels) if label == 1][:int(args.ratio * count_minority)])

            texts = balanced_texts
            labels = balanced_labels

    number_of_classes = len(set(labels))

    if number_of_classes > 2:
        labels = [label - 1 for label in labels]

    print(
        f'data loaded successfully with {len(texts)} rows and {number_of_classes} labels')

    sample_weights = get_sample_weights(labels)

    return texts, labels, number_of_classes, sample_weights


class MyDataset(Dataset):
    def __init__(self, texts, labels, args):
        self.texts = texts
        self.labels = labels
        self.length = len(self.texts)

        self.vocabulary = args.alphabet + args.extra_characters
        self.number_of_characters = args.number_of_characters + \
            len(args.extra_characters)
        self.max_length = args.max_length
        self.preprocessing_steps = args.steps
        self.identity_mat = np.identity(self.number_of_characters)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.texts[index]

        data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text)[::-1] if i in self.vocabulary],
                        dtype=np.float32)
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data), self.number_of_characters), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros(
                (self.max_length, self.number_of_characters), dtype=np.float32)

        label = self.labels[index]
        data = torch.Tensor(data)

        return data, label
