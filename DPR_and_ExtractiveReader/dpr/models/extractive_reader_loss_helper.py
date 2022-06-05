import torch

INF = 1e10

def document_level_loss(start_logits, start_positions_list,
                        end_logits, end_positions_list,
                        answer_positions_mask,
                        global_loss, null_ans_index=0,
                        global_steps=-1, anneal_steps=-1):
    # Here, we assume the null answer has index 0.
    not_null_ans = torch.max(torch.logical_or(start_positions_list > null_ans_index, end_positions_list > null_ans_index).float(), dim=2, keepdim=True)[0]
    doc_answer_positions_mask = not_null_ans * answer_positions_mask
    positive_par_mask = torch.max(doc_answer_positions_mask, dim=2)[0]

    if global_loss == 'h2-pos-mml':
        # H2 document-level position-based MML loss.
        total_loss = doc_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            positive_par_mask, loss_type='h2_mml'
        )
    elif global_loss == 'h2-pos-hard_em':
        # H2 document-level position-based HardEM loss.
        total_loss = doc_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            positive_par_mask,
            loss_type='h2_hard_em'
        )
    elif global_loss == 'h2-pos-hard_em_anneal':
        # H2 document-level position-based HardEM with annealing.
        is_warmup = float(global_steps < anneal_steps)
        total_loss = (1.0 - is_warmup) * doc_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            positive_par_mask, loss_type='h2_hard_em'
        ) + is_warmup * doc_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            positive_par_mask, loss_type='h2_mml'
        )
    elif global_loss == 'h2-span-mml':
        # H2 document-level span-based MML loss.
        total_loss = doc_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            positive_par_mask, loss_type='h2_mml'
        )
    elif global_loss == 'h2-span-hard_em':
        # H2 document-level span-based HardEM loss.
        total_loss = doc_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            positive_par_mask, loss_type='h2_hard_em'
        )
    elif global_loss == 'h2-span-hard_em_anneal':
        # H2 document-level span-based HardEM with annealing.
        is_warmup = float(global_steps < anneal_steps)
        total_loss = (1.0 - is_warmup) * doc_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            positive_par_mask, loss_type='h2_hard_em'
        ) + is_warmup * doc_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask,
            positive_par_mask, loss_type='h2_mml'
        )
    elif global_loss == 'h3-span-mml':
        # H3 document-level span-based MML loss.
        total_loss = doc_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, doc_answer_positions_mask,
            positive_par_mask, loss_type='h3_mml'
        )
    elif global_loss == 'h3-span-hard_em':
        # H3 document-level span-based HardEM loss.
        total_loss = doc_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, doc_answer_positions_mask,
            positive_par_mask, loss_type='h3_hard_em'
        )
    elif global_loss == 'h3-span-hard_em_anneal':
        # H3 docuemnt-level span-based HardEM with annealing:
        is_warmup = float(global_steps < anneal_steps)
        total_loss = (1.0 - is_warmup) * doc_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, doc_answer_positions_mask,
            positive_par_mask, loss_type='h3_hard_em'
        ) + is_warmup * doc_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, doc_answer_positions_mask,
            positive_par_mask, loss_type='h3_mml'
        )
    elif global_loss == 'h3-pos-mml':
        # H3 document-level position-based MML loss.
        total_loss = doc_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, doc_answer_positions_mask,
            positive_par_mask, loss_type='h3_mml'
        )
    elif global_loss == 'h3-pos-hard_em':
        # H3 document-level position-based HardEM loss.
        total_loss = doc_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, doc_answer_positions_mask,
            positive_par_mask, loss_type='h3_hard_em'
        )
    elif global_loss == 'h3-pos-hard_em_anneal':
        # H3 docuemnt-level position-based hardEM with annealing.
        is_warmup = float(global_steps < anneal_steps)
        total_loss = (1.0 - is_warmup) * doc_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, doc_answer_positions_mask,
            positive_par_mask, loss_type='h3_hard_em'
        ) + is_warmup * doc_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, doc_answer_positions_mask,
            positive_par_mask, loss_type='h3_mml'
        )
    elif global_loss == 'h1':
        # H1 document-level loss.
        total_loss = doc_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, doc_answer_positions_mask,
            positive_par_mask, loss_type='h1'
        )
    else:
        raise ValueError("Unknown global loss %s" % global_loss)

    return total_loss


def paragraph_level_loss(start_logits, start_positions_list,
                                 end_logits, end_positions_list,
                                 answer_positions_mask, local_loss,
                                 global_steps=-1, anneal_steps=-1):
    has_ans_par_mask = torch.max(answer_positions_mask, dim=2)[0]
    if local_loss  == 'h2-pos-mml':
        # H2 paragraph-level position-based MML loss.
        total_loss = par_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask, has_ans_par_mask,
            loss_type='h2_mml',
        )
    elif local_loss == 'h2-span-mml':
        # H2 paragraph-level span-based MML loss.
        total_loss = par_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask, has_ans_par_mask,
            loss_type='h2_mml',
        )
    elif local_loss == 'h2-pos-hard_em':
        # H2 paragraph-level pos-based HardEM loss.
        total_loss = par_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask, has_ans_par_mask,
            loss_type='h2_hard_em',
        )
    elif local_loss == 'h2-pos-hard_em_anneal':
        # H2 paragraph-level pos-based HardEM loss with annealing.
        is_warmup = float(global_steps < anneal_steps)
        total_loss = (1.0 - is_warmup) * par_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask, has_ans_par_mask,
            loss_type='h2_hard_em',
        ) + is_warmup * par_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask, has_ans_par_mask,
            loss_type='h2_mml',
        )
    elif local_loss == 'h2-span-hard_em':
        # Paragraph-level span-based marginalization loss.
        total_loss = par_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask, has_ans_par_mask,
            loss_type='h2_hard_em',
        )
    elif local_loss == 'h2-span-hard_em_anneal':
        # H2 paragraph-level span-based HardEM loss with annealing.
        is_warmup = float(global_steps < anneal_steps)
        total_loss = (1.0 - is_warmup) * par_span_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask, has_ans_par_mask,
            loss_type='h2_hard_em',
        ) + is_warmup * par_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask, has_ans_par_mask,
            loss_type='h2_mml',
        )
    elif local_loss == 'h1':
        # Paragraph-level H1 loss.
        total_loss = par_pos_loss(
            start_logits, start_positions_list, end_logits,
            end_positions_list, answer_positions_mask, has_ans_par_mask,
            loss_type='h1'
        )
    else:
        raise ValueError("Unknown local_loss %s" % local_loss)

    return total_loss


def compute_span_log_score(start_log_scores, start_pos_list, end_log_scores, end_pos_list):
    """Computes the span log scores."""
    ans_span_start_log_scores = torch.gather(start_log_scores, 2, start_pos_list)
    ans_span_end_log_scores = torch.gather(end_log_scores, 2, end_pos_list)
    return (ans_span_start_log_scores + ans_span_end_log_scores)


def compute_logprob(logits, dim):
    """Computes the log prob based on logits."""
    return logits - torch.logsumexp(logits, dim=dim, keepdim=True)


def doc_span_loss(start_logits, start_indices, end_logits, end_indices,
                  positions_mask, pos_par_mask, loss_type=None):
    """Computes document-level normalization span-based losses.
    Arguments:
        * start_logits (torch.FloatTensor): [batch_size, max_passages, max_seq_len]
        * start_indices (torch.LongTensor): [batch_size, max_passages, max_spans]
        * end_logits (torch.FloatTensor): [batch_size, max_passages, max_seq_len]
        * end_indices (torch.LongTensor): [batch_size, max_passages, max_spans]
        * positions_mask (torch.FloatTensor): [batch_size, max_passages, max_seq_len]
        * pos_par_mask (torch.FloatTensor): [batch_size, max_passages]
        * loss_type (str)
    """
    batch_size = start_logits.size(0)
    # Computes the log prob for start and end positions.
    start_log_prob = compute_logprob(start_logits, dim=[1, 2])
    end_log_prob = compute_logprob(end_logits, dim=[1, 2])

    # Computes the log prob for a span, which the sum of the corresponding start
    # position log prob and the end position log prob.
    span_log_prob = compute_span_log_score(
        start_log_prob, start_indices, end_log_prob, end_indices
    )

    log_score_mask = positions_mask.float()
    pos_par_mask = pos_par_mask.float()
    has_positives = (torch.sum(pos_par_mask, dim=1) > 0.5).float()

    if loss_type == "h2_mml":
        # Each positive paragraph contains a correct span.
        span_loss = -torch.sum(pos_par_mask * torch.logsumexp(span_log_prob + (log_score_mask - 1) * INF, dim=2, keepdim=False))
    elif loss_type == "h2_hard_em":
        span_loss = -torch.sum(pos_par_mask * torch.max(span_log_prob + (log_score_mask - 1) * INF, dim=2)[0])
    elif loss_type == "h3_mml":
        # The whole document contains one correct span.
        span_loss = -torch.sum(has_positives * torch.logsumexp(span_log_prob + (log_score_mask - 1) * INF, dim=[1, 2]))
    elif loss_type == "h3_hard_em":
        span_loss = -torch.sum(has_positives * torch.max((span_log_prob + (log_score_mask - 1) * INF).view(batch_size, -1), dim=1)[0])
    else:
        raise ValueError("Unknwon loss_type %s for doc_span_loss!" % loss_type)

    return span_loss / batch_size


def one_hot_answer_positions(position_list, position_mask, depth):
    position_tensor = torch.zeros((position_list.size(0), position_list.size(1), depth), device=position_list.device, dtype=torch.float32)
    position_tensor.scatter_(2, position_list, 1.0)
    position_tensor = (position_list > 0.5).float() * position_mask.unsqueeze(2)
    return position_tensor


def compute_masked_log_score(log_score, position_list, answer_masks):
    position_tensor = one_hot_answer_positions(position_list, answer_masks, log_score.size(2))
    return log_score + (position_tensor - 1) * INF


def doc_pos_loss(start_logits, start_indices, end_logits, end_indices,
                 answer_positions_mask, pos_par_mask, loss_type=None):
    """Computes document-level normalization position-based losses.
    Arguments:
        * start_logits (torch.FloatTensor): [batch_size, max_passages, max_seq]
        * start_indices (torch.LongTensor): [batch_size, max_passages, max_spans]
        * end_logits (torch.FloatTensor): [batch_size, max_passages, max_seq]
        * end_indices (torch.LongTensor): [batch_size, max_passages, max_spans]
        * answer_positions_mask (torch.FloatTensor): [batch_size, max_passages, max_spans]
        * pos_par_mask (torch.FloatTensor): [batch_size, max_passages]
    """
    batch_size = start_logits.size(0)
    # Computes the log prob for start and end positions.
    start_log_prob = compute_logprob(start_logits, dim=[1, 2])
    end_log_prob = compute_logprob(end_logits, dim=[1, 2])

    answer_positions_mask = answer_positions_mask.float()
    pos_par_mask = pos_par_mask.float()
    has_positives = (torch.sum(pos_par_mask, dim=1) > 0.5).float()

    masked_start_log_prob = compute_masked_log_score(
        start_log_prob, start_indices, answer_positions_mask,
    )
    masked_end_log_prob = compute_masked_log_score(
        end_log_prob, end_indices, answer_positions_mask,
    )

    if loss_type == "h2_mml":
        # Each positive paragraph contains a correct span.
        start_loss = -torch.sum(pos_par_mask * torch.logsumexp(masked_start_log_prob, dim=2, keepdim=False))
        end_loss = -torch.sum(pos_par_mask * torch.logsumexp(masked_end_log_prob, dim=2, keepdim=False))
    elif loss_type == "h2_hard_em":
        # Each positive paragraph contains a correct span.
        start_loss = -torch.sum(pos_par_mask * torch.max(masked_start_log_prob, dim=2)[0])
        end_loss = -torch.sum(pos_par_mask * torch.max(masked_end_log_prob, dim=2)[0])
    elif loss_type == "h3_mml":
        # The whole document contains one correct span.
        start_loss = -torch.sum(has_positives * torch.logsumexp(masked_start_log_prob, dim=[1, 2]))
        end_loss = -torch.sum(has_positives * torch.logsumexp(masked_end_log_prob, dim=[1, 2]))
    elif loss_type == "h3_hard_em":
        # The whole document contains one correct span.
        start_loss = -torch.sum(has_positives * torch.max(masked_start_log_prob.view(batch_size, -1), dim=1)[0])
        end_loss = -torch.sum(has_positives * torch.max(masked_end_log_prob.view(batch_size, -1), dim=1)[0])
    elif loss_type == "h1":
        # All spans are correct
        start_position_tensor = one_hot_answer_positions(
            start_indices, answer_positions_mask, start_logits.size(2)
        )
        end_position_tensor = one_hot_answer_positions(
            end_indices, answer_positions_mask, end_logits.size(2)
        )

        start_loss = -torch.sum(start_position_tensor * start_log_prob)
        end_loss = -torch.sum(end_position_tensor * end_log_prob)
    else:
        raise ValueError("Unknwon loss_type %s for doc_pos_loss!" % loss_type)

    return (start_loss + end_loss) / 2.0 / batch_size


def par_span_loss(start_logits, start_indices, end_logits, end_indices,
                  positions_mask, has_ans_par_mask, loss_type=None):
    """Computes paragraph-level normalization span-based losses."""
    batch_size = start_logits.size(0)
    # Computes the log prob for start and end positions.
    start_log_prob = compute_logprob(start_logits, dim=2)
    end_log_prob = compute_logprob(end_logits, dim=2)

    # Computes the log prob for a span, which the sum of the corresponding start
    # position log prob and the end position log prob.
    span_log_prob = compute_span_log_score(
        start_log_prob, start_indices, end_log_prob, end_indices
    )

    positions_mask = positions_mask.float()
    has_ans_par_mask = has_ans_par_mask.float()
    masked_span_log_prob = span_log_prob + (positions_mask - 1) * INF

    if loss_type == "h2_mml":
        # Each positive paragraph contains a correct span.
        span_loss = -torch.sum(has_ans_par_mask * torch.logsumexp(masked_span_log_prob, dim=2))
    elif loss_type == "h2_hard_em":
        span_loss = -torch.sum(has_ans_par_mask * torch.max(masked_span_log_prob, dim=2)[0])
    elif loss_type == "h1":
        span_loss = -torch.sum(positions_mask * masked_span_log_prob)
    else:
        raise ValueError("Unknwon loss_type %s for par_span_loss!" % loss_type)

    return span_loss / batch_size


def par_pos_loss(start_logits, start_indices, end_logits, end_indices,
                 answer_positions_mask, has_ans_par_mask, loss_type=None):
    """Computes paragraph-level normalization position-based losses."""
    batch_size = start_logits.size(0)
    # Computes the log prob for start and end positions.
    start_log_prob = compute_logprob(start_logits, dim=2)
    end_log_prob = compute_logprob(end_logits, dim=2)

    answer_positions_mask = answer_positions_mask.float()
    has_ans_par_mask = has_ans_par_mask.float()

    masked_start_log_prob = compute_masked_log_score(
        start_log_prob, start_indices, answer_positions_mask,
    )

    masked_end_log_prob = compute_masked_log_score(
        end_log_prob, end_indices, answer_positions_mask,
    )

    if loss_type == "h2_mml":
        # Each positive paragraph contains a correct span.
        start_loss = -torch.sum(has_ans_par_mask * torch.logsumexp(masked_start_log_prob, dim=2))
        end_loss = -torch.sum(has_ans_par_mask * torch.logsumexp(masked_end_log_prob, dim=2))
    elif loss_type == "h2_hard_em":
        # Each positive paragraph contains a correct span.
        start_loss = -torch.sum(has_ans_par_mask * torch.max(masked_start_log_prob, dim=2)[0])
        end_loss = -torch.sum(has_ans_par_mask * torch.max(masked_end_log_prob, dim=2)[0])
    elif loss_type == "h1":
        # All spans are correct
        start_position_tensor = one_hot_answer_positions(
            start_indices, answer_positions_mask, start_logits.size(2)
        )
        end_position_tensor = one_hot_answer_positions(
            end_indices, answer_positions_mask, end_logits.size(2)
        )
        start_loss = -torch.sum(start_position_tensor * masked_start_log_prob)
        end_loss = -torch.sum(end_position_tensor * masked_end_log_prob,)
    else:
        raise ValueError("Unknwon loss_type %s for par_pos_loss!" % loss_type)

    return (start_loss + end_loss) / 2.0 / batch_size
