#encoding:utf-8
import torch

def prepare_pack_padded_sequence(inputs_words,
                                 seq_lengths,
                                 descending=True):
    """
    :param use_cuda:
    :param inputs_words:
    :param seq_lengths:
    :param descending:
    :return:
    """
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=descending)
    _, desorted_indices = torch.sort(indices, descending=False)
    sorted_inputs_words = inputs_words[indices]
    return sorted_inputs_words, sorted_seq_lengths, desorted_indices
