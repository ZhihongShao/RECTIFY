#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Set of utilities for Q&A results validation tasks - Retriver passage validation and Reader predicted answer validation
"""

import collections
import logging
import string
import unicodedata
from multiprocessing import Pool as ProcessPool

import regex as re
import regex
from functools import partial
from typing import Tuple, List, Dict

import numpy as np
import argparse
import gzip
import json

from tqdm import tqdm

QAMatchStats = collections.namedtuple(
    "QAMatchStats", ["top_k_hits", "questions_doc_hits"]
)

QATableMatchStats = collections.namedtuple(
    "QAMatchStats", ["top_k_chunk_hits", "top_k_table_hits", "questions_doc_hits"]
)

class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.

        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups

class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()


class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        # if len(kwargs.get('annotators', {})) > 0:
        #     logger.warning('%s only tokenizes! Skipping annotators: %s' %
        #                    (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
            ))
        return Tokens(data, self.annotators)


def calculate_matches(
    samples: List[Dict],
    workers_num: int,
    doc_ids=None,
    predictions=None,
    match_type: str = 'string',
    verbose: bool = True,
) -> QAMatchStats:
    """
    Evaluates answers presence in the set of documents. This function is supposed to be used with a large collection of
    documents and results. It internally forks multiple sub-processes for evaluation and then merges results
    :param all_docs: dictionary of the entire documents database. doc_id -> (doc_text, title)
    :param answers: list of answers's list. One list per question
    :param closest_docs: document ids of the top results along with their scores
    :param workers_num: amount of parallel threads to process data
    :param match_type: type of answer matching. Refer to has_answer code for available options
    :return: matching information tuple.
    top_k_hits - a list where the index is the amount of top documents retrieved and the value is the total amount of
    valid matches across an entire dataset.
    questions_doc_hits - more detailed info with answer matches for every question and every retrieved document
    """
    n_docs = [1, 5, 10, 15, 20, 30, 40, 100, 200, 300, 400, 500]

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    if verbose:
        print("Matching answers in top docs...")
    get_score_partial = partial(
        check_answer, match_type=match_type, tokenizer=tokenizer, n_docs=n_docs
    )

    inputs = [{'sample': sample} for sample in samples]
    if doc_ids is not None:
        for inp, _doc_ids in zip(inputs, doc_ids):
            inp['doc_ids'] = _doc_ids
    if predictions is not None:
        for inp, _pred in zip(inputs, predictions):
            inp['predictions'] = _pred

    if workers_num > 1:
        processes = ProcessPool(processes=workers_num)
        scores = processes.map(get_score_partial, inputs)
    else:
        scores = list(map(get_score_partial, inputs))

    if verbose:
        print("Per question validation results len={}".format(len(scores)))

    results = {}
    tot_all = set()
    tot_multi = set()
    for key in ['all', 'multi']:
        for n_doc in n_docs:
            hits = 0
            tot = 0
            for ret in scores:
                if n_doc in ret and key in ret[n_doc]:
                    hits += ret[n_doc][key]
                    tot += 1
            if tot:
                if key == 'all':
                    tot_all.add(tot)
                else:
                    tot_multi.add(tot)
            results["{}-MRECALL@{}".format(key, n_doc)] = hits / max(tot, 1)
        for ans_metric in ['f1', 'precision', 'recall']:
            lst = [ret['{}-ans-{}'.format(key, ans_metric)] for ret in scores if '{}-ans-{}'.format(key, ans_metric) in ret]
            results['{}-ans-{}'.format(key, ans_metric)] = float(np.mean(lst)) if lst else 0.0
    if len(tot_multi) == 0:
        tot_multi.add(0)
    assert len(tot_all) == 1
    assert len(tot_multi) == 1
    results['n_samples'] = list(tot_all)[0]
    results['n_samples (multi)'] = list(tot_multi)[0]
    if verbose:
        print(results)
    return results


def check_answer(inp, tokenizer, match_type, n_docs) -> Dict:
    """Search through all the top docs to see if they have any of the answers."""
    sample = inp['sample']
    doc_ids = inp.get('doc_ids', range(len(sample['ctxs'])))
    preds = inp.get('predictions', None)

    ret = {}

    if 'annotations' not in sample:
        assert 'answers' in sample
        sample['annotations'] = [
            {
                'type': 'singleAnswer',
                'answer': sample['answers'],
            }
        ]
    is_multi = not any(anno['type'] == 'singleAnswer' for anno in sample['annotations'])
    for annotation in sample['annotations']:
        if annotation['type'] == 'singleAnswer':
            answers = [annotation['answer']]
        else:
            assert annotation['type'] == 'multipleQAs'
            answers = [answer['answer'] for answer in annotation['qaPairs']]
        ans_found = np.zeros((len(answers), ), dtype=np.int32)
        for i, doc_id in enumerate(doc_ids):
            doc = sample['ctxs'][doc_id]
            text = doc['text']
            for j, ans in enumerate(answers):
                if ans_found[j] == 0 and has_answer(ans, text, tokenizer, match_type):
                    ans_found[j] = 1
            if i + 1 in n_docs:
                if i + 1 not in ret:
                    ret[i + 1] = {
                        'all': 0
                    }
                ret[i + 1]['all'] = max(ret[i + 1]['all'], np.sum(ans_found) >= min(i + 1, len(answers)))
                if is_multi:
                    ret[i + 1]['multi'] = max(ret[i + 1].get('multi', 0), np.sum(ans_found) >= min(i + 1, len(answers)))
        if preds is not None:
            for key, val in zip(['f1', 'precision', 'recall'], get_f1_prec_recall(answers, preds)):
                ret['all-ans-{}'.format(key)] = max(ret.get('all-ans-{}'.format(key), 0), val)
                if is_multi:
                    ret['multi-ans-{}'.format(key)] = max(ret.get('multi-ans-{}'.format(key), 0), val)
    return ret


def has_answer(answers, text, tokenizer, match_type) -> bool:
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    text = _normalize(text)

    if match_type == "string":
        # Answer is a list of possible strings
        text = tokenizer.tokenize(text).words(uncased=True)

        for single_answer in answers:
            single_answer = _normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)

            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i : i + len(single_answer)]:
                    return True

    elif match_type == "regex":
        # Answer is a regex
        for single_answer in answers:
            single_answer = _normalize(single_answer)
            if regex_match(text, single_answer):
                return True
    return False


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(pattern, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException:
        return False
    return pattern.search(text) is not None

def _normalize(text):
    return unicodedata.normalize("NFD", text)

# function for the reader model answer validation
def exact_match_score(prediction, ground_truth):
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)

def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_exact_match(answers1, answers2):
    if type(answers1)==list:
        if len(answers1)==0:
            return 0
        return np.max([get_exact_match(a, answers2) for a in answers1])
    if type(answers2)==list:
        if len(answers2)==0:
            return 0
        return np.max([get_exact_match(answers1, a) for a in answers2])
    return (_normalize_answer(answers1) == _normalize_answer(answers2))


def get_f1_prec_recall(answers, predictions, is_equal=get_exact_match):
    '''
    :answers: a list of list of strings
    :predictions: a list of strings
    '''
    assert len(answers)>0 and len(predictions)>0, (answers, predictions)
    occupied_answers = [False for _ in answers]
    occupied_predictions = [False for _ in predictions]
    for i, answer in enumerate(answers):
        for j, prediction in enumerate(predictions):
            if occupied_answers[i] or occupied_predictions[j]:
                continue
            em = is_equal(answer, prediction)
            if em:
                occupied_answers[i] = True
                occupied_predictions[j] = True
    assert np.sum(occupied_answers)==np.sum(occupied_predictions)
    a, b = np.mean(occupied_answers), np.mean(occupied_predictions)
    if a+b==0:
        return 0, 0, 0
    return 2*a*b/(a+b), b, a

def read_wiki_split(filename):
    id2doc = {}
    with gzip.open(filename, "rb") as file:
        file.readline()
        for line in tqdm(file, desc='Reading wiki splits'):
            line = line.decode().split("\t")
            id2doc[int(line[0])] = line[1:]
    return id2doc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_filename", type=str, default="")
    parser.add_argument("--rerank_filename", type=str, default="")
    parser.add_argument("--reader_score_filename", type=str, default="")
    args = parser.parse_args()

    samples = json.load(open(args.src_filename, "r"))
    if args.rerank_filename:
        doc_ids = []
        answers = []
        with open(args.rerank_filename, "r", encoding='utf-8') as file:
            for line in file:
                line = json.loads(line)
                indices = sorted(line['indices'], key=lambda x: -x[-1])
                doc_ids.append([item[0] for item in indices])
                answers.append(line['answers'])
    elif args.reader_score_filename:
        doc_ids = []
        reader_scores = json.load(open(args.reader_score_filename, "r"))
        for scores in reader_scores:
            doc_ids.append(list(reversed(np.argsort([item['score'] for item in scores['ctxs']]))))
            answers = None
    else:
        doc_ids = None
        answers = None
    calculate_matches(samples, 16, doc_ids, answers)

if __name__ == '__main__':
    main()
