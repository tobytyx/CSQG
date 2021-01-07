# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional
from models.attention import SessionAttention
from utils.constants import max_generate_length
from utils.Beam import Beam, get_inst_idx_to_tensor_position_map
from utils.Beam import collate_active_info, collect_hypothesis_and_scores
from utils import constants
from icecream import ic


class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding, n_layers,
                 p, bidirectional, out_p, device):
        super(GRUEncoder, self).__init__()
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(out_p) if out_p > 0 else None
        self.num_layer = n_layers
        self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, dropout=p, bidirectional=bidirectional)
        self.device = device

    def forward(self, input_seq, lengths, padding_value=0):
        """
        :param input_seq:
        :param lengths:
        :param padding_value:
        :return outputs: B * len * hidden, hidden: B * hidden
        """
        if self.embedding:
            input_seq = self.embedding(input_seq)
        sorted_seq_lengths, indices = torch.sort(lengths, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)  # 还原需要的indices
        input_seq = input_seq.index_select(0, indices)
        packed = nn.utils.rnn.pack_padded_sequence(input_seq, sorted_seq_lengths, batch_first=True)
        # Forward pass through RNN cell
        outputs, hidden = self.rnn(packed)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, padding_value=padding_value)
        outputs = outputs[desorted_indices]
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        hidden = hidden.transpose(0, 1)[desorted_indices]
        if self.dropout:
            outputs = self.dropout(outputs)
            hidden = self.dropout(hidden)
        return outputs, hidden[:, -1, :]


class GRUEncoderAtt(nn.Module):
    def __init__(self, input_size, hidden_size, attn, visual_compress,
                 glimpse, n_layers, p, out_p, bidirectional, device):
        super(GRUEncoderAtt, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(out_p) if out_p > 0 else None
        self.num_layer = n_layers
        self.input_size = input_size
        self.rnn = nn.GRU(input_size, hidden_size, n_layers,
                          batch_first=False, dropout=p, bidirectional=bidirectional)
        self.attn = attn
        self.visual_compress = visual_compress
        self.glimpse = glimpse
        self.device = device

    def forward(self, input_seq, lengths, v_feature):
        """
        :param input_seq: B * max_len * hidden
        :param lengths: B
        :param v_feature: B * num * v_dim
        """
        bs, max_len = input_seq.size(0), input_seq.size(1)
        input_seq = input_seq.transpose(1, 0)
        dtype = input_seq.dtype
        outputs = []
        h = None
        for i in range(max_len):
            mask = (lengths > i).to(torch.long)
            active, non_active = torch.nonzero(mask).squeeze(-1), torch.nonzero(1 - mask).squeeze(-1)
            indices = torch.cat([active, non_active])
            _, recover_indices = torch.sort(indices, dim=0)
            ac_token = torch.index_select(input_seq[i], 0, active)
            if h is not None:
                ac_h = torch.index_select(h, 1, active)
            else:
                ac_h = None
            ac_v = torch.index_select(v_feature, 0, active)
            ac_o, ac_h = self.gru_step(ac_token, ac_h, ac_v)
            # renew h
            pad_h = torch.zeros((self.num_layer, non_active.size(0), self.hidden_size), dtype=dtype, device=self.device)
            h = torch.index_select(torch.cat([ac_h, pad_h], dim=1), 1, recover_indices)
            # renew out
            pad_o = torch.zeros((non_active.size(0), self.hidden_size), dtype=dtype, device=self.device)
            o = torch.index_select(torch.cat([ac_o, pad_o], dim=0), 0, recover_indices)
            outputs.append(o)
        outputs = torch.stack(outputs, 0).transpose(1, 0)
        return outputs, h[-1]

    def gru_step(self, input_token, h, v_feature):
        """
        :param input_token: b * hidden
        :param h: num_layer * b * hidden
        :param v_feature: b * num * v_dim
        """
        v_num = v_feature.size(1)
        if self.attn is not None and h is not None:
            att_scores = self.attn(v_feature, h[-1])
        else:
            att_scores = torch.ones((v_feature.size(0), v_num), device=self.device) / v_num
        v_feature = torch.einsum('bn,bnv->bv', [att_scores, v_feature])
        v_feature = self.visual_compress(v_feature.unsqueeze(2)).squeeze(2)
        rnn_input = torch.cat([input_token, v_feature], dim=1)
        rnn_input = rnn_input.unsqueeze(0)
        output, h_n = self.rnn(rnn_input, h)
        output = output.squeeze(0)
        if self.dropout:
            output = self.dropout(output)
            h_n = self.dropout(h_n)
        return output, h_n


class GRUDecoderAtt(nn.Module):
    def __init__(self, n_vocab, embedding, embedding_dim, category_dim, session_dim, v_feature_dim, n_layers,
                 hidden_size, attention_size, dropout, beam, device):
        super(GRUDecoderAtt, self).__init__()
        # Define parameters
        self.n_vocab = n_vocab
        self.hidden_size = hidden_size
        self.n_bm = beam
        self.use_visual = True if v_feature_dim is not None else False
        rnn_input_dim = session_dim + embedding_dim + category_dim
        if self.use_visual:
            rnn_input_dim += v_feature_dim
        # Define layers
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        self.attn = SessionAttention(hidden_size, session_dim, attention_size)
        self.n_layers = n_layers
        self.rnn = nn.GRU(rnn_input_dim, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.out = nn.Linear(hidden_size, n_vocab)
        torch.nn.init.normal_(self.out.weight, 0.0, 0.1)
        self.device = device

    def forward(self, tgt, decoder_state, category, v_feature, session_outputs, session_mask):
        """
        :param tgt: B * max_len
        :param decoder_state: B * decoder_hidden
        :param category: B x category_dim
        :param v_feature: B x v_feature_dim / None
        :param session_outputs: B x encoder_len x session_hidden
        :param session_mask: B x max_turn
        :return:
        """
        batch_size, tgt_len = tgt.size()
        tgt_embedded = self.embedding(tgt)
        tgt_embedded = tgt_embedded.transpose(1, 0)  # len x B x d_word_vec
        outputs = []
        vocab_outputs = []
        h = decoder_state.repeat(self.n_layers, 1).view(batch_size, self.n_layers, -1)
        h = h.transpose(1, 0).contiguous()
        for t in range(tgt_len):
            word_input = tgt_embedded[t]
            vocab_output, output, h = self.decoder_step(h, word_input, category, v_feature, session_outputs, session_mask)
            outputs.append(output)
            vocab_outputs.append(vocab_output)
        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.transpose(1, 0)  # B * len * hidden
        vocab_outputs = torch.stack(vocab_outputs, dim=0)
        vocab_outputs = vocab_outputs.transpose(1, 0)  # B * len * hidden
        return vocab_outputs, outputs

    def decoder_step(self, h, last_output, category, v_feature, session_outputs, session_mask, embed=False):
        """

        :param h: num_layer x B x hidden
        :param last_output: B x hidden
        :param category: B x category_dim
        :param v_feature: B x v_feature_dim / None
        :param session_outputs: B x len x session_hidden
        :param session_mask: session pad mask B x len
        :param embed
        :return:
        """
        if embed:
            last_output = self.embedding(last_output)
        if h is not None:
            session_scores = self.attn(h[-1], session_outputs, session_mask)
        else:
            bs, s_len, _ = session_outputs.size()
            session_scores = torch.ones(bs, s_len, device=self.device) / (bs * s_len)
        session = torch.einsum("bl,blh->bh", [session_scores, session_outputs])
        if self.use_visual and v_feature is not None:
            rnn_input = torch.cat((last_output, session, category, v_feature), dim=1).unsqueeze(1)
        else:
            rnn_input = torch.cat((last_output, session, category), dim=1).unsqueeze(1)
        output, h = self.rnn(rnn_input, h)
        output = output.squeeze(1)
        vocab_output = self.out(output)
        return vocab_output, output, h

    def generate(self, decoder_state, category, v_feature, session_outputs, session_mask):
        """
        :param decoder_state:
        :param category: B x category_dim
        :param v_feature: B x v_feature_size / None
        :param session_outputs: B x session_len x session_hidden
        :param session_mask: B x session_len
        :return:
        """

        # -- Repeat data for beam search
        n_inst, encoder_len, s_h = session_outputs.size()
        category = category.repeat(1, self.n_bm).view(n_inst * self.n_bm, -1)
        session_outputs = session_outputs.repeat(1, self.n_bm, 1).view(n_inst * self.n_bm, encoder_len, s_h)
        session_mask = session_mask.repeat(1, self.n_bm).view(n_inst * self.n_bm, -1)
        # -- Prepare beams
        inst_dec_beams = [Beam(self.n_bm, device=self.device) for _ in range(n_inst)]

        # -- Bookkeeping for active or not
        active_inst_idx_list = list(range(n_inst))
        inst_idx_to_pos_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        # -- Decode
        h = decoder_state.repeat(self.n_layers, 1).view(n_inst, self.n_layers, -1)
        h = h.repeat(1, self.n_bm, 1).view(n_inst * self.n_bm, self.n_layers, -1)
        for len_dec_seq in range(1, max_generate_length + 1):
            active_inst_idx_list, h = self.beam_decode_step(
                inst_dec_beams, len_dec_seq, h, category, v_feature, session_outputs, session_mask,
                inst_idx_to_pos_map
            )
            if not active_inst_idx_list:
                break  # all instances have finished their path to <EOS>
            if self.use_visual and v_feature is not None:
                h, category, v_feature, session_outputs, session_mask, inst_idx_to_pos_map = collate_active_info(
                    h, category, v_feature, session_outputs, session_mask,
                    inst_idx_to_position_map=inst_idx_to_pos_map, active_inst_idx_list=active_inst_idx_list,
                    device=self.device, n_bm=self.n_bm
                )
            else:
                h, category, session_outputs, session_mask, inst_idx_to_pos_map = collate_active_info(
                    h, category, session_outputs, session_mask,
                    inst_idx_to_position_map=inst_idx_to_pos_map, active_inst_idx_list=active_inst_idx_list,
                    device=self.device, n_bm=self.n_bm
                )

        batch_hyp, _ = collect_hypothesis_and_scores(inst_dec_beams, self.n_bm)
        result = []
        for hyps in batch_hyp:
            result.append(hyps[0])
        return result

    def beam_decode_step(self, inst_dec_beams, len_dec_seq, h, c, v, enc_out, mask, inst_idx_to_position_map):
        n_active_inst = len(inst_idx_to_position_map)
        # prepare_beam_dec_seq
        dec_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_seq = torch.stack(dec_seq).to(self.device)
        dec_seq = dec_seq.view(-1, len_dec_seq)
        # predict_word
        if h is not None:
            h = h.transpose(1, 0).contiguous()
        dec_vocab_output, dec_output, h = self.decoder_step(h, dec_seq[:, -1], c, v, enc_out, mask, embed=True)
        h = h.transpose(1, 0).contiguous()
        word_prob = nn.functional.softmax(dec_vocab_output, dim=1)
        word_prob = word_prob.view(n_active_inst, self.n_bm, -1)

        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            is_inst_complete = inst_dec_beams[inst_idx].advance(word_prob[inst_position])
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]
        return active_inst_idx_list, h


class GRUDecoderBase(nn.Module):
    def __init__(self, n_vocab, embedding, embedding_dim, category_dim, n_layers, hidden_size, dropout, beam, device):
        super(GRUDecoderBase, self).__init__()
        # Define parameters
        self.n_vocab = n_vocab
        self.hidden_size = hidden_size
        self.n_bm = beam
        # Define layers
        self.embedding = embedding
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.category_dim = category_dim
        rnn_input_dim = embedding_dim
        if category_dim is not None:
            rnn_input_dim += category_dim
        self.rnn = nn.GRU(rnn_input_dim, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.out = nn.Linear(hidden_size, n_vocab)
        torch.nn.init.normal_(self.out.weight, 0.0, 0.1)
        self.device = device

    def forward(self, tgt, decoder_state, category=None):
        """
        :param tgt: B * max_len
        :param decoder_state: B * hidden
        :param category: B * category_dim / None
        :return:
        """
        batch_size, tgt_len = tgt.size()
        tgt_embedded = self.embedding(tgt)
        tgt_embedded = tgt_embedded.transpose(1, 0)  # len x B x d_word_vec
        outputs = []
        vocab_outputs = []
        h = decoder_state.repeat(self.n_layers, 1).view(batch_size, self.n_layers, -1)
        h = h.transpose(1, 0).contiguous()
        for t in range(tgt_len):
            word_input = tgt_embedded[t]
            vocab_out, output, h = self.decoder_step(h, category, word_input)
            outputs.append(output)
            vocab_outputs.append(vocab_out)
        outputs = torch.stack(outputs, dim=0)
        vocab_outputs = torch.stack(vocab_outputs, dim=0)
        outputs = outputs.transpose(1, 0)  # B * len * hidden
        vocab_outputs = vocab_outputs.transpose(1, 0)  # B * len * vocab_size
        return vocab_outputs, outputs

    def decoder_step(self, h, category, last_output, embed=False):
        """
        :param h: num_layer x B x hidden
        :param category: B x category_dim / None
        :param last_output: B x hidden
        :param embed
        :return:
        """
        if embed:
            last_output = self.embedding(last_output)
        if category is not None:
            last_output = torch.cat([last_output, category], dim=-1)
        last_output = last_output.unsqueeze(1)
        output, h = self.rnn(last_output, h)
        output = output.squeeze(1)
        vocab_out = self.out(output)
        return vocab_out, output, h

    def greedy_search_generate(self, decoder_state, category=None):
        bsz = decoder_state.size(0)
        outputs = []
        vocab_outputs = []
        dtype = decoder_state.dtype
        h = decoder_state.repeat(self.n_layers, 1).view(bsz, self.n_layers, -1)
        h = h.transpose(1, 0).contiguous()
        last_output = torch.tensor([constants.SOS] * bsz, dtype=torch.long).to(self.device)
        for i in range(max_generate_length):
            mask = last_output.ne(constants.EOS).to(torch.long)
            if torch.sum(mask) == 0:
                break
            active, non_active = torch.nonzero(mask).squeeze(-1), torch.nonzero(1 - mask).squeeze(-1)
            indices = torch.cat([active, non_active])
            _, recover_indices = torch.sort(indices, dim=0)
            ac_token = torch.index_select(last_output, 0, active)
            ac_h = torch.index_select(h, 1, active)
            ac_category = torch.index_select(category, 0, active)
            ac_vocab_out, ac_o, ac_h = self.decoder_step(ac_h, ac_category, ac_token, True)
            # renew h
            pad_h = torch.zeros((self.n_layers, non_active.size(0), self.hidden_size), dtype=dtype, device=self.device)
            h = torch.index_select(torch.cat([ac_h, pad_h], dim=1), 1, recover_indices)
            # renew out
            pad_o = torch.zeros((non_active.size(0), self.hidden_size), dtype=dtype, device=self.device)
            o = torch.index_select(torch.cat([ac_o, pad_o], dim=0), 0, recover_indices)
            pad_vocab_out = torch.zeros((non_active.size(0), self.n_vocab), dtype=dtype, device=self.device)
            pad_vocab_out[:, 0] = 1
            vocab_out = torch.index_select(torch.cat([ac_vocab_out, pad_vocab_out], dim=0), 0, recover_indices)
            last_output = torch.argmax(ac_vocab_out, dim=1).to(torch.long)
            pad_last_output = torch.tensor([constants.EOS] * non_active.size(0), dtype=torch.long, device=self.device)
            last_output = torch.index_select(torch.cat([last_output, pad_last_output], dim=0), 0, recover_indices)
            outputs.append(o)
            vocab_outputs.append(vocab_out)

        outputs = torch.stack(outputs, 0).transpose(1, 0)
        vocab_outputs = torch.stack(vocab_outputs, 0).transpose(1, 0)
        return vocab_outputs, outputs

    def generate(self, decoder_state, category=None):
        """
        :param decoder_state: B * hidden
        :param category: B * category
        :return:
        """
        # -- Repeat data for beam search
        n_inst, s_h = decoder_state.size()
        h = decoder_state.repeat(self.n_layers, 1).view(n_inst, self.n_layers, s_h)
        h = h.repeat(1, self.n_bm, 1).view(n_inst * self.n_bm, self.n_layers, s_h)
        if category is not None:
            category = category.repeat(1, self.n_bm).view(n_inst * self.n_bm, -1)
        # -- Prepare beams
        inst_dec_beams = [Beam(self.n_bm, device=self.device) for _ in range(n_inst)]

        # -- Bookkeeping for active or not
        active_inst_idx_list = list(range(n_inst))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        # -- Decode
        for len_dec_seq in range(1, max_generate_length + 1):
            active_inst_idx_list, h = self.beam_decode_step(
                inst_dec_beams, len_dec_seq, h, category, inst_idx_to_position_map
            )
            if not active_inst_idx_list:
                break  # all instances have finished their path to <EOS>
            if category is not None:
                h, category, inst_idx_to_position_map = collate_active_info(
                    h, category,
                    inst_idx_to_position_map=inst_idx_to_position_map, active_inst_idx_list=active_inst_idx_list,
                    device=self.device, n_bm=self.n_bm
                )
            else:
                h, inst_idx_to_position_map = collate_active_info(
                    h, inst_idx_to_position_map=inst_idx_to_position_map, active_inst_idx_list=active_inst_idx_list,
                    device=self.device, n_bm=self.n_bm
                )
        batch_hyp, _ = collect_hypothesis_and_scores(inst_dec_beams, self.n_bm)
        result = []
        for hyps in batch_hyp:
            result.append(hyps[0])
        return result

    def beam_decode_step(self, inst_dec_beams, len_dec_seq, h, category, inst_idx_to_position_map):
        n_active_inst = len(inst_idx_to_position_map)
        # prepare_beam_dec_seq
        dec_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_seq = torch.stack(dec_seq).to(self.device)
        dec_seq = dec_seq.view(-1, len_dec_seq)
        # predict_word
        if h is not None:
            h = h.transpose(1, 0).contiguous()
        dec_vocab_output, dec_output, h = self.decoder_step(h, category, dec_seq[:, -1], embed=True)
        h = h.transpose(1, 0).contiguous()
        word_prob = nn.functional.softmax(dec_vocab_output, dim=1)
        word_prob = word_prob.view(n_active_inst, self.n_bm, -1)

        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            is_inst_complete = inst_dec_beams[inst_idx].advance(word_prob[inst_position])
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]
        return active_inst_idx_list, h


class GRUDecoderCategoryOnce(GRUDecoderBase):
    def __init__(self, n_vocab, embedding, embedding_dim, category_dim, n_layers, hidden_size, dropout, beam, device):
        super(GRUDecoderCategoryOnce, self).__init__(
            n_vocab, embedding, embedding_dim, category_dim, n_layers, hidden_size, dropout, beam, device)
        # Define parameters
        self.hidden_size = hidden_size if category_dim is None else hidden_size + category_dim
        # Define layers
        self.embedding = embedding
        self.rnn = nn.GRU(embedding_dim, self.hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.out = nn.Linear(self.hidden_size, n_vocab)
        torch.nn.init.normal_(self.out.weight, 0.0, 0.1)

    def forward(self, tgt, decoder_state, category=None):
        """
        :param tgt: B * max_len
        :param decoder_state: B * hidden
        :param category: B * category_dim / None
        :return:
        """
        batch_size, tgt_len = tgt.size()
        tgt_embedded = self.embedding(tgt)
        tgt_embedded = tgt_embedded.transpose(1, 0)  # len x B x d_word_vec
        outputs = []
        vocab_outputs = []
        if category is not None:
            decoder_state = torch.cat([decoder_state, category], dim=1)
        h = decoder_state.repeat(self.n_layers, 1).view(batch_size, self.n_layers, -1)
        h = h.transpose(1, 0).contiguous()  # B * (hidden + category)
        for t in range(tgt_len):
            word_input = tgt_embedded[t]
            vocab_output, output, h = self.decoder_step(h, category, word_input)
            outputs.append(output)
            vocab_outputs.append(vocab_output)
        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.transpose(1, 0)  # B * len * hidden
        vocab_outputs = torch.stack(vocab_outputs, dim=0)
        vocab_outputs = vocab_outputs.transpose(1, 0)  # B * len * hidden
        return vocab_outputs, outputs

    def decoder_step(self, h, category, last_output, embed=False):
        """
        :param h: num_layer x B x hidden
        :param category: B x category_dim / None
        :param last_output: B x hidden
        :param embed
        :return:
        """
        if embed:
            last_output = self.embedding(last_output)
        # if category is not None:
        #     last_output = torch.cat([last_output, category], dim=-1)
        last_output = last_output.unsqueeze(1)
        output, h = self.rnn(last_output, h)
        output = output.squeeze(1)
        vocab_out = self.out(output)
        return vocab_out, output, h

    def generate(self, decoder_state, category=None):
        """
        :param decoder_state: B * hidden
        :param category: B * category
        :return:
        """

        # -- Repeat data for beam search
        if category is not None:
            decoder_state = torch.cat([decoder_state, category], dim=1)
        n_inst, s_h = decoder_state.size()
        h = decoder_state.repeat(self.n_layers, 1).view(n_inst, self.n_layers, s_h)
        h = h.repeat(1, self.n_bm, 1).view(n_inst * self.n_bm, self.n_layers, s_h)
        # -- Prepare beams
        inst_dec_beams = [Beam(self.n_bm, device=self.device) for _ in range(n_inst)]

        # -- Bookkeeping for active or not
        active_inst_idx_list = list(range(n_inst))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        # -- Decode
        for len_dec_seq in range(1, max_generate_length + 1):
            active_inst_idx_list, h = self.beam_decode_step(
                inst_dec_beams, len_dec_seq, h, None, inst_idx_to_position_map
            )
            if not active_inst_idx_list:
                break  # all instances have finished their path to <EOS>
            h, inst_idx_to_position_map = collate_active_info(
                h, inst_idx_to_position_map=inst_idx_to_position_map, active_inst_idx_list=active_inst_idx_list,
                device=self.device, n_bm=self.n_bm
            )
        batch_hyp, _ = collect_hypothesis_and_scores(inst_dec_beams, self.n_bm)
        result = []
        for hyps in batch_hyp:
            result.append(hyps[0])
        return result
