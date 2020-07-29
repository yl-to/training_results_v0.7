# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import numpy as np
import os

from torch.utils.data import ConcatDataset

from fairseq import options
from fairseq.data import (
    data_utils, Dictionary, LanguagePairDataset,
    IndexedRawTextDataset,
)

from fairseq.data.indexed_dataset import IndexedRawTokenIDDataset
from fairseq.data.indexed_dataset import IndexedInMemoryDataset
from fairseq.data.indexed_dataset import MockedInMemoryDataset

from . import FairseqTask, register_task


@register_task('translation')
class TranslationTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC', help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET', help='target language')
        parser.add_argument('--raw-text', action='store_true', help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL', help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL', help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=256, type=int, metavar='N', help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=256, type=int, metavar='N', help='max number of tokens in the target sequence')
        parser.add_argument('--seq-len-multiple', default=1, type=int, metavar='N', help='Pad sequences to a multiple of N')

        parser.add_argument('--uniform-n-seq-per-batch', default=None, type=int, metavar='N', help='Make uniform batches with this many sequences')
        parser.add_argument('--uniform-seq-len-per-batch', default=None, type=int, metavar='N', help='Make uniform batches with this seq len')
        parser.add_argument('--uniform-n-seq-in-dataset', default=None, type=int, metavar='N', help='If creating uniform batches with mock data, this is the dataset size')

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data)
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))
        
        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, combine=False):
        """Load a dataset split."""

        def split_exists(split, src, tgt, lang):
            filename = os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            print('filename:', filename)
            print('raw_text:', self.args.raw_text)
            if self.args.raw_text and IndexedRawTokenIDDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedInMemoryDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text and not self.args.uniform_n_seq_per_batch and not self.args.uniform_seq_len_per_batch:
                return IndexedRawTokenIDDataset(path, dictionary)
            elif IndexedInMemoryDataset.exists(path) and not self.args.uniform_n_seq_per_batch and not self.args.uniform_seq_len_per_batch:
                return IndexedInMemoryDataset(path)
            elif self.args.uniform_n_seq_per_batch and self.args.uniform_seq_len_per_batch:
                if self.args.uniform_n_seq_in_dataset:
                    return MockedInMemoryDataset(path, self.args.uniform_n_seq_in_dataset, self.args.uniform_n_seq_per_batch, self.args.uniform_seq_len_per_batch)
            return None

        src_datasets = []
        tgt_datasets = []

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')

            # infer langcode
            src, tgt = self.args.source_lang, self.args.target_lang
            if split_exists(split_k, src, tgt, src):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split_k, src, tgt))
            elif split_exists(split_k, tgt, src, src):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split_k, tgt, src))
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))

            src_datasets.append(indexed_dataset(prefix + src, self.src_dict))
            tgt_datasets.append(indexed_dataset(prefix + tgt, self.tgt_dict))

            print('| {} {} {} examples'.format(self.args.data, split_k, len(src_datasets[-1])))

            if not combine:
                break

        assert len(src_datasets) == len(tgt_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
            src_sizes = src_dataset.sizes
            tgt_sizes = tgt_dataset.sizes
        else:
            src_dataset = ConcatDataset(src_datasets)
            tgt_dataset = ConcatDataset(tgt_datasets)
            src_sizes = np.concatenate([ds.sizes for ds in src_datasets])
            tgt_sizes = np.concatenate([ds.sizes for ds in tgt_datasets])

        print('srcline:', src_dataset[0])

        self.datasets[split] = LanguagePairDataset(
            src_dataset, src_sizes, self.src_dict,
            tgt_dataset, tgt_sizes, self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            seq_len_multiple=self.args.seq_len_multiple,
        )

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict