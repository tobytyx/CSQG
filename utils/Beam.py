# -*- coding: utf-8 -*-
""" Manage beam search info structure.

    Heavily borrowed from OpenNMT-py.
    For code in OpenNMT-py, please check the following link:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""
import torch
from utils import constants


class Beam(object):
    def __init__(self, size, device):

        self.size = size
        self._done = False

        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float).to(device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), constants.PAD, dtype=torch.long).to(device)]
        self.next_ys[0][0] = constants.SOS

    def get_current_state(self):
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        """
        Update beam status and check if finished or not.
        word_prob: beam_size * vocab_size
        """
        num_words = word_prob.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)

        # best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)  # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)  # 2nd sort
        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].item() == constants.EOS:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[constants.SOS] + h for h in hyps]
            dec_seq = torch.tensor(hyps, dtype=torch.long)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))


def get_inst_idx_to_tensor_position_map(inst_idx_list):
    """ Indicate the position of an instance in a tensor. """
    return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}


def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
    """ Collect tensor parts associated to active instances. """
    _, *d_hs = beamed_tensor.size()
    n_curr_active_inst = curr_active_inst_idx.size(0)
    new_shape = (n_curr_active_inst * n_bm, *d_hs)
    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
    beamed_tensor = beamed_tensor.view(*new_shape)
    return beamed_tensor


def collect_hypothesis_and_scores(inst_dec_beams, n_best):
    all_hyp, all_scores = [], []
    for inst_idx in range(len(inst_dec_beams)):
        scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
        all_scores += [scores[:n_best]]

        hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
        hyps = [hyp for hyp in hyps if hyp != constants.EOS]
        all_hyp += [hyps]
    return all_hyp, all_scores


def collate_active_info(*infos, inst_idx_to_position_map, active_inst_idx_list, n_bm, device):
    # Sentences which are still active are collected,
    # so the decoder will not run on completed sentences.
    n_prev_active_inst = len(inst_idx_to_position_map)
    active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
    active_inst_idx = torch.tensor(active_inst_idx, dtype=torch.long, device=device)
    active_infos = []
    for info in infos:
        active_info = collect_active_part(info, active_inst_idx, n_prev_active_inst, n_bm)
        active_infos.append(active_info)
    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
    active_infos.append(active_inst_idx_to_position_map)
    return active_infos
