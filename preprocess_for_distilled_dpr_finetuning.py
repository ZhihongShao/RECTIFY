import argparse
import os
import json
import regex as re
import regex
from functools import partial
from typing import Tuple, List, Dict
import multiprocessing
from multiprocessing import Pool
import numpy as np
import gzip
import collections
from itertools import chain
import logging
import string
import unicodedata
from tqdm import tqdm
from copy import deepcopy

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


def _normalize(text):
    return unicodedata.normalize("NFD", text)


def process_ambigqa_instance(sample, is_train, tokenizer):
    n_context = 1000
    num_positives = 6
    num_hard_negatives = 30
    num_other_negatives = 50

    q2a = {}
    if is_train:
        for anno in sample['annotations']:
            if anno['type'] == 'singleAnswer':
                q = anno['type'].lower()
                if q not in q2a:
                    q2a[q] = set()
                for a in anno['answer']:
                    q2a[q].add(a)
            else:
                for qa in anno['qaPairs']:
                    q = qa['question'].lower()
                    if q not in q2a:
                        q2a[q] = set()
                    for a in qa['answer']:
                        q2a[q].add(a)
    else:
        answers = set()
        for anno in sample['annotations']:
            if anno['type'] == 'singleAnswer':
                answers.update(anno['answer'])
            else:
                for qa in anno['qaPairs']:
                    answers.update(qa['answer'])
        q2a[sample['question']] = answers
    ret = []
    for _, answers in q2a.items():
        answers = list(answers)
        positive_ctxs = []
        negative_ctxs = []
        hard_negative_ctxs = []
        if is_train:
            for cid, ctx in enumerate(sample['ctxs'][:n_context]):
                if ctx['hasanswer'] and cid < 100 and len(positive_ctxs) < num_positives and has_answer(answers, ctx['text'], tokenizer, 'string'):
                    positive_ctxs.append(ctx)
                elif not ctx['hasanswer']:
                    if len(hard_negative_ctxs) < num_hard_negatives:
                        hard_negative_ctxs.append(ctx)
                    else:
                        negative_ctxs.append(ctx)
            if is_train and (not positive_ctxs or not negative_ctxs or not hard_negative_ctxs):
                continue
            if negative_ctxs:
                indices = np.random.choice(list(range(len(negative_ctxs))), num_other_negatives)
                negative_ctxs = [negative_ctxs[i] for i in indices]
        else:
            positive_ctxs = sample['ctxs'][:n_context]
        item = {
            key: sample[key] for key in ['question', 'id', 'annotations']
        }
        item.update({
            'answers': answers,
            'positive_ctxs': positive_ctxs,
            'negative_ctxs': negative_ctxs,
            'hard_negative_ctxs': hard_negative_ctxs,
        })
        ret.append(item)
    return ret


def process_webqsp_instance(sample, is_train, tokenizer):
    n_context = 1000
    num_positives = 6
    num_hard_negatives = 30
    num_other_negatives = 50

    q2a = {}
    if is_train:
        num_answer = 0
        for anno in sample['annotations']:
            if anno['type'] == 'singleAnswer':
                q = str(num_answer)
                num_answer += 1
                if q not in q2a:
                    q2a[q] = set()
                for a in anno['answer']:
                    q2a[q].add(a)
            else:
                for qa in anno['qaPairs']:
                    q = str(num_answer)
                    num_answer += 1
                    if q not in q2a:
                        q2a[q] = set()
                    for a in qa['answer']:
                        q2a[q].add(a)
    else:
        answers = set()
        for anno in sample['annotations']:
            if anno['type'] == 'singleAnswer':
                answers.update(anno['answer'])
            else:
                for qa in anno['qaPairs']:
                    answers.update(qa['answer'])
        q2a[sample['question']] = answers
    ret = []
    for _, answers in q2a.items():
        answers = list(answers)
        positive_ctxs = []
        other_positive_ctxs = []
        negative_ctxs = []
        hard_negative_ctxs = []
        if is_train:
            for cid, ctx in enumerate(sample['ctxs'][:n_context]):
                num_positives = 6 if cid < 100 else 3
                if ctx['hasanswer'] and cid < 500 and len(positive_ctxs) < num_positives and has_answer(answers, ctx['text'], tokenizer, 'string'):
                    if cid < 100:
                        positive_ctxs.append(ctx)
                    else:
                        other_positive_ctxs.append(ctx)
                elif not ctx['hasanswer']:
                    if len(hard_negative_ctxs) < num_hard_negatives:
                        hard_negative_ctxs.append(ctx)
                    else:
                        negative_ctxs.append(ctx)
            if is_train and (not positive_ctxs or not negative_ctxs or not hard_negative_ctxs):
                continue
            if negative_ctxs:
                indices = np.random.choice(list(range(len(negative_ctxs))), num_other_negatives)
                negative_ctxs = [negative_ctxs[i] for i in indices]
        else:
            positive_ctxs = sample['ctxs'][:n_context]
        item = {
            key: sample[key] for key in ['question', 'id', 'annotations']
        }
        item.update({
            'answers': answers,
            'positive_ctxs': positive_ctxs,
            'other_positive_ctxs': other_positive_ctxs,
            'negative_ctxs': negative_ctxs,
            'hard_negative_ctxs': hard_negative_ctxs,
        })
        ret.append(item)
    return ret


def process_dataset(process_fn, tasks, is_train, num_workers):
    tokenizer = SimpleTokenizer()
    process_fn = partial(
        process_fn, is_train=is_train, tokenizer=tokenizer,
    )
    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()
    if num_workers > 1:
        pool = Pool(num_workers)
        return list(chain(*list(tqdm(pool.imap(process_fn, tasks, chunksize=20), desc='preprocessing', total=len(tasks)))))
    else:
        return list(chain(*[process_fn(task) for task in tqdm(tasks, desc='preprocessing', total=len(tasks))]))


def stat_processed_data(data):
    print("total size: {}".format(len(data)))
    q2n = {}
    for item in data:
        q = item['question'].lower()
        q2n[q] = q2n.get(q, 0) + 1
    lst = sorted(list(q2n.values()))
    print("num distinct questions: {}".format(len(q2n)))
    print("mean num instances per question: {}".format(np.mean(lst)))
    print("var of num instances per question: {}".format(np.var(lst)))
    for ratio in [0.25, 0.5, 0.75, 0.9]:
        print("percentile-{}% num instances per question: {}".format(ratio * 100, lst[int(len(lst) * ratio)]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--out_file", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=-1)
    args = parser.parse_args()

    if 'ambigqa' in args.dataset.lower():
        tasks = json.load(open(args.dataset, "r"))
        is_train = 'train' in args.dataset.lower()
        data = process_dataset(process_ambigqa_instance, tasks, is_train, args.num_workers)
    elif 'webqsp' in args.dataset.lower():
        tasks = json.load(open(args.dataset, "r"))
        is_train = 'train' in args.dataset.lower()
        data = process_dataset(process_webqsp_instance, tasks, is_train, args.num_workers)
    else:
        raise NotImplementedError()

    stat_processed_data(data)
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    json.dump(data, open(args.out_file, "w"))
    print("data saved to {}".format(args.out_file), flush=True)

if __name__ == '__main__':
    main()
