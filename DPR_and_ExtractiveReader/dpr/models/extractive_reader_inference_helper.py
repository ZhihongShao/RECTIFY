import collections
from collections import defaultdict
import numpy as np
from dpr.data.extractive_reader_data import (
    ReaderPassage,
    ReaderSample,
    SpanPrediction,
    get_best_spans,
)

import torch

ReaderQuestionPredictions = collections.namedtuple(
    "ReaderQuestionPredictions", ["id", "predictions", "gold_answers"]
)

def _get_best_spans_from_passage(
    tensorizer,
    reader_passage: ReaderPassage,
    start_logits,
    end_logits,
    relevance_score,
    max_answer_length,
):
    sequence_ids = reader_passage.sequence_ids
    sequence_len = sequence_ids.size(0)
    # assuming question & title information is at the beginning of the sequence
    passage_offset = reader_passage.passage_offset

    p_start_logits = start_logits.tolist()[
        passage_offset:sequence_len
    ]
    p_end_logits = end_logits.tolist()[
        passage_offset:sequence_len
    ]

    ctx_ids = sequence_ids.tolist()[passage_offset:]
    if reader_passage.token_is_max_context:
        token_is_max_context = defaultdict(lambda: False)
        for key in reader_passage.token_is_max_context.keys():
            token_is_max_context[key - passage_offset] = reader_passage.token_is_max_context[key]
    else:
        token_is_max_context = reader_passage.token_is_max_context
    best_spans = get_best_spans(
        tensorizer,
        p_start_logits,
        p_end_logits,
        ctx_ids,
        max_answer_length,
        reader_passage.passage_rank,
        reader_passage.sub_doc_rank,
        relevance_score,
        -1,
        token_is_max_context,
        top_spans=20,
    )
    return best_spans

def rerank_based_max_inference(
    tensorizer,
    start_logits,
    end_logits,
    relevance_logits,
    samples_batch: List[ReaderSample],
    max_answer_length: int,
    n_best_size: int = 1,
    passage_thresholds: List[int] = None,
    allow_null_answer: bool = False,
    null_score_thres: float = 0.5,
) -> List[ReaderQuestionPredictions]:

    questions_num, passages_per_question = relevance_logits.size()

    _, idxs = torch.sort(
        relevance_logits,
        dim=1,
        descending=True,
    )

    batch_results = []
    for q in range(questions_num):
        sample = samples_batch[q]

        non_empty_passages_num = len(sample.passages)
        nbest = []
        for p in range(passages_per_question):
            passage_idx = idxs[q, p].item()
            if (
                passage_idx >= non_empty_passages_num
            ):  # empty passage selected, skip
                continue

            best_spans = _get_best_spans_from_passage(
                tensorizer,
                sample.passages[passage_idx],
                start_logits[q, passage_idx],
                end_logits[q, passage_idx],
                relevance_logits[q, passage_idx].item(),
                max_answer_length,
            )
            for span in best_spans:
                span.score = span.span_score
            nbest.extend(best_spans)
            if len(nbest) > 0 and not passage_thresholds:
                break

        n_eval_passages = passage_thresholds if passage_thresholds is not None else [max(passage.passage_rank for passage in sample.passages) + 1]
        predictions = {}
        for n in n_eval_passages:
            curr_nbest = [pred for pred in nbest if pred.passage_index < n]
            if allow_null_answer and \
                (
                    not curr_nbest or \
                    torch.sigmoid(max(pred.relevance_score for pred in curr_nbest)) < null_score_thres
                ):
                span = SpanPrediction()
                span.relevance_score = max(pred.relevance_score for pred in curr_nbest)
                predictions[n] = [span]
                continue

            if curr_nbest:
                predictions[n] = curr_nbest[:n_best_size]
        batch_results.append(
            ReaderQuestionPredictions(sample.question, predictions, sample.answers)
        )
    return batch_results


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _get_doc_norm(start_logits, end_logits):
    return torch.logsumexp(start_logits) + torch.logsumexp(end_logits)


def max_inference(
    tensorizer,
    samples_start_logits,
    samples_end_logits,
    samples_batch: List[ReaderSample],
    max_answer_length: int,
    n_best_size: int = 1,
    passage_thresholds: List[int] = None,
    allow_null_answer: bool = False,
    null_score_diff: float = 0.0,
) -> List[ReaderQuestionPredictions]:
    return _n_best_inference(
        tensorizer,
        samples_start_logits,
        samples_end_logits,
        samples_batch,
        max_answer_length,
        'max',
        n_best_size,
        passage_thresholds,
        allow_null_answer,
        null_score_diff
    )


def sum_inference(
    tensorizer,
    samples_start_logits,
    samples_end_logits,
    samples_batch: List[ReaderSample],
    max_answer_length: int,
    n_best_size: int = 1,
    passage_thresholds: List[int] = None,
    allow_null_answer: bool = False,
    null_score_diff: float = 0.0,
) -> List[ReaderQuestionPredictions]:
    return _n_best_inference(
        tensorizer,
        samples_start_logits,
        samples_end_logits,
        samples_batch,
        max_answer_length,
        'max',
        n_best_size,
        passage_thresholds,
        allow_null_answer,
        null_score_diff
    )


def _n_best_inference(
    tensorizer,
    samples_start_logits,
    samples_end_logits,
    samples_batch: List[ReaderSample],
    max_answer_length: int,
    infer_type: str,
    n_best_size: int = 1,
    passage_thresholds: List[int] = None,
    allow_null_answer: bool = False,
    null_score_diff: float = 0.0,
) -> List[ReaderQuestionPredictions]:

    batch_results = []
    for example_index, (example, start_logits, end_logits) in enumerate(zip(samples_batch, samples_start_logits, samples_end_logits)):
        features = example.passages

        nbest = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature = None  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        # compute doc norm
        num_passage2doc_norm = defaultdict(lambda: -1)

        for num_features, (feature, subdoc_start_logits, subdoc_end_logits) in enumerate(zip(features + [None], start_logits + [None], end_logits + [None])):
            if passage_thresholds is not None:
                passage_rank = feature.passage_rank if feature is not None else features[-1].passage_rank + 1
                sub_doc_rank = feature.sub_doc_rank if feature is not None else -1
                if passage_rank in passage_thresholds and sub_doc_rank <= 0:
                    num_passage2doc_norm[passage_rank] = _get_doc_norm(start_logits[:num_features], end_logits[:num_features])

                    if allow_null_answer:
                        nbest.append(
                            SpanPrediction("", null_start_logit, null_end_logit, score_null, -1, -1, min_null_feature.passage_rank, min_null_feature.sub_doc_rank)
                        )

            if feature is not None:
                best_spans = _get_best_spans_from_passage(
                    tensorizer,
                    feature,
                    subdoc_start_logits,
                    subdoc_end_logits,
                    -1,
                    max_answer_length,
                )
                nbest.extend(best_spans)

                # if we could have irrelevant answers, get the min score of irrelevant
                if allow_null_answer:
                    subdoc_start_logits = subdoc_start_logits.tolist()
                    subdoc_end_logits = subdoc_end_logits.tolist()
                    feature_null_score = subdoc_start_logits[0] + subdoc_end_logits[0]
                    if feature_null_score < score_null:
                        score_null = feature_null_score
                        min_null_feature = feature
                        null_start_logit = subdoc_start_logits[0]
                        null_end_logit = subdoc_end_logits[0]

        nbest = sorted(nbest, key=lambda x: x.span_score, reverse=True)

        n_eval_passages = passage_thresholds if passage_thresholds is not None else [max(passage.passage_rank for passage in features) + 1]

        predictions = {}
        for n in n_eval_passages:
            curr_nbest = [pred for pred in nbest if pred.passage_rank < n]
            doc_norm = num_passage2doc_norm[n]

            if allow_null_answer and \
                (
                    not curr_nbest or \
                    curr_nbest[0].text == "" and \
                    (
                        len(curr_nbest) == 1 or \
                        curr_nbest[0].span_score - curr_nbest[1].start_logit - curr_nbest[1].end_logit > null_score_diff
                    )
                ):
                span = SpanPrediction()
                span.score = float(np.exp(curr_nbest[0].span_score - doc_norm))
                predictions[n] = [span]
                continue

            ans2preds = {}
            ans2score = {}
            for pred in curr_nbest:
                ans = pred.prediction_text
                if ans not in ans2preds:
                    ans2preds[ans] = []
                    ans2score[ans] = 0 if infer_type == 'sum' else -1e10
                ans2preds[ans].append(pred)
                normalized_prob = float(np.exp(pred.span_score - doc_norm))
                if infer_type == 'sum':
                    ans2score[ans] += normalized_prob
                else:
                    ans2score[ans] = max(ans2score[ans], normalized_prob)
            if ans2score:
                ans2score = sorted(list(ans2score.items()), reverse=True)
                tmp = []
                for ans, score in ans2score[:n_best_size]:
                    span = SpanPrediction()
                    span.prediction_text = ans
                    span.score = score
                    tmp.append(tmp)
                predictions[n] = tmp

        batch_results.append(
            ReaderQuestionPredictions(example.question, predictions, example.answers)
        )
    return batch_results
