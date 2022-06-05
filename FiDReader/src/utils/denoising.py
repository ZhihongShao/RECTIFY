import functools
import numpy as np
import torch
import torch_scatter

def span_corruption(tokens, vocab_size, mean_noise_span_length=3.0, noise_density=0.15):
    """Final pretraining objective used in Raffel et al., 2019."""
    return denoise(
        tokens,
        inputs_fn=functools.partial(
            noise_span_to_unique_sentinel,
            vocab_size=vocab_size
        ),
        targets_fn=functools.partial(
            nonnoise_span_to_unique_sentinel,
            vocab_size=vocab_size
        ),
        noise_density=noise_density,
        noise_mask_fn=functools.partial(
            random_spans_noise_mask,
            mean_noise_span_length=mean_noise_span_length
        )
    )

def denoise(tokens,
            noise_density,
            noise_mask_fn,
            inputs_fn,
            targets_fn=None):
    """Gin-configurable token preprocessor for self-supervised denoising tasks.
    This function takes a dataset containing "targets" sequences,
    and turns each sequence into a dictionary containing:
    {
        "inputs": noisy version of the original sequence
        "targets": the full original sequence or missing parts of original sequence
    }
    """
    noise_mask = noise_mask_fn(tokens.size(0), noise_density)
    inputs = inputs_fn(tokens, noise_mask)
    if targets_fn:
        targets = targets_fn(tokens, noise_mask)
    else:
        targets = tokens
    return inputs, targets

def noise_span_to_unique_sentinel(tokens, noise_mask, vocab_size):
    """Replace each run of consecutive noise tokens with a different sentinel.
    The idea here is to be able to align the dropped spans in the inputs
    with the markers in the targets.
    We want to generate training examples like
    "We hold X to be Y that" -> "X these truths Y self evident Z"
    Sentinels assigned in decreasing order within the sequence starting at
    vocabulary.size - 1.  That is, we appropriate the last tokens in the
    vocabulary for additional use as sentinels.
    TODO(noam): we may want to try enlarging the vocabulary and leaving room
    for the sentinels instead.  However, this requires enlarging the embedding
    tables in the model, so that is a bigger change.
    Args:
        tokens: a 1d integer Tensor
        noise_mask: a boolean Tensor with the same shape as tokens
        vocabulary: a vocabulary.Vocabulary
    Returns:
        a Tensor with the same shape and dtype as tokens
    """

    prev_token_is_noise = torch.cat((torch.zeros((1, ), dtype=noise_mask.dtype, device=noise_mask.device), noise_mask[:-1]), 0)

    first_noise_tokens = torch.logical_and(
        noise_mask, torch.logical_not(prev_token_is_noise))
    subsequent_noise_tokens = torch.logical_and(noise_mask, prev_token_is_noise)

    sentinel = vocab_size - torch.cumsum(first_noise_tokens.to(tokens.dtype), 0)

    tokens = torch.where(first_noise_tokens, sentinel, tokens)
    return torch.masked_select(tokens, torch.logical_not(subsequent_noise_tokens))

def nonnoise_span_to_unique_sentinel(tokens, noise_mask, vocab_size):
    return noise_span_to_unique_sentinel(tokens, torch.logical_not(noise_mask), vocab_size)

def stateless_shuffle(value):
  """Randomly shuffles a tensor, statelessly."""
  flat_value = value.view(-1)
  
  indices = torch.argsort(
      torch.rand(flat_value.shape)
  )
  flat_shuffle = torch.gather(flat_value, 0, indices)
  return flat_shuffle.view(value.shape)

def random_spans_noise_mask(length,
                            noise_density,
                            mean_noise_span_length=3.0):
    """Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(
        num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
        length: an int32 scalar (length of the incoming token sequence)
        noise_density: a float - approximate density of output mask
        seeds: an int32 Tensor, shaped (2, 2)
        mean_noise_span_length: a number
    Returns:
        a boolean tensor with shape [length]
    """

    orig_length = length
    # increase length to avoid degeneracy
    length = max(length, 2)
    num_noise_tokens = int(round(float(length) * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(round(float(num_noise_tokens) / mean_noise_span_length))
    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens
    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
        num_items: an integer scalar > 0
        num_segments: an integer scalar in [1, num_items]
        seed: an integer seed
        Returns:
        a Tensor with shape [num_segments] containing positive integers that add
        up to num_items
        """
        first_in_segment = torch.cat(
            (
                torch.zeros((1,), dtype=torch.int32),
                stateless_shuffle((torch.arange(num_items - 1) < num_segments - 1).to(torch.int32)),
            ),
            0
        )
        segment_id = torch.cumsum(first_in_segment, 0)
        segment_length = torch_scatter.scatter(torch.ones_like(segment_id), segment_id)
        return segment_length
    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)
    interleaved_span_lengths = torch.stack([nonnoise_span_lengths, noise_span_lengths], axis=1).view(num_noise_spans * 2)
    span_starts = torch.cumsum(interleaved_span_lengths, 0)[:-1]
    span_start_indicator = torch.zeros((length,), dtype=span_starts.dtype)
    span_start_indicator.scatter_(0, span_starts, torch.ones_like(span_starts))
    span_num = torch.cumsum(span_start_indicator, 0)
    is_noise = (span_num % 2 == 1)
    num_nonnoise_end_tokens = np.random.choice(span_starts[0])
    is_noise = torch.cat((is_noise[num_nonnoise_end_tokens:], is_noise[:num_nonnoise_end_tokens]), 0)
    return is_noise[:orig_length]

if __name__ == '__main__':
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained("/home/shaozhihong/.cache/torch/t5-base")
    tokens = torch.tensor(tokenizer.encode("What does marthur mean in the last name?", add_special_tokens=False))
    for i in range(10):
        inputs, targets = span_corruption(tokens, tokenizer.vocab_size)
        print(tokenizer.decode(inputs), "\t", tokenizer.decode(targets))
