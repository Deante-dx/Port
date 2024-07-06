import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)


class WordEmbedding(nn.Module):
    def __init__(self, num_words, word_dim, drop_rate, word_vectors=None):
        super(WordEmbedding, self).__init__()
        self.is_pretrained = False if word_vectors is None else True
        if self.is_pretrained:
            self.pad_vec = nn.Parameter(torch.zeros(size=(1, word_dim), dtype=torch.float32), requires_grad=False)
            unk_vec = torch.empty(size=(1, word_dim), requires_grad=True, dtype=torch.float32)
            nn.init.xavier_uniform_(unk_vec)
            self.unk_vec = nn.Parameter(unk_vec, requires_grad=True)
            self.glove_vec = nn.Parameter(torch.tensor(word_vectors, dtype=torch.float32), requires_grad=False)
        else:
            self.word_emb = nn.Embedding(num_words, word_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, word_ids):
        if self.is_pretrained:
            word_emb = F.embedding(word_ids, torch.cat([self.pad_vec, self.unk_vec, self.glove_vec], dim=0),
                                   padding_idx=0)
        else:
            word_emb = self.word_emb(word_ids)
        return self.dropout(word_emb)


class CharacterEmbedding(nn.Module):
    def __init__(self, num_chars, char_dim, drop_rate):
        super(CharacterEmbedding, self).__init__()
        self.char_emb = nn.Embedding(num_chars, char_dim, padding_idx=0)
        kernels, channels = [1, 2, 3, 4], [10, 20, 30, 40]
        self.char_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=char_dim, out_channels=channel, kernel_size=(1, kernel), stride=(1, 1), padding=0,
                          bias=True),
                nn.ReLU()
            ) for kernel, channel in zip(kernels, channels)
        ])
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, char_ids):
        char_emb = self.char_emb(char_ids)  # (batch_size, w_seq_len, c_seq_len, char_dim)
        char_emb = self.dropout(char_emb)
        char_emb = char_emb.permute(0, 3, 1, 2)  # (batch_size, char_dim, w_seq_len, c_seq_len)
        char_outputs = []
        for conv_layer in self.char_convs:
            output = conv_layer(char_emb)
            output, _ = torch.max(output, dim=3, keepdim=False)  # reduce max (batch_size, channel, w_seq_len)
            char_outputs.append(output)
        char_output = torch.cat(char_outputs, dim=1)  # (batch_size, sum(channels), w_seq_len)
        return char_output.permute(0, 2, 1)  # (batch_size, w_seq_len, sum(channels))


class Embedding(nn.Module):
    def __init__(self, num_words, num_chars, word_dim, char_dim, drop_rate, out_dim, word_vectors=None):
        super(Embedding, self).__init__()
        self.word_emb = WordEmbedding(num_words, word_dim, drop_rate, word_vectors=word_vectors)
        self.char_emb = CharacterEmbedding(num_chars, char_dim, drop_rate)
        # output linear layer
        # self.linear = Conv1D(in_dim=word_dim + 100, out_dim=out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.linear = Conv1D(in_dim=word_dim, out_dim=out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.mask_embedding = nn.Embedding(1, out_dim)

    def forward(self, word_ids, char_ids, word_masks):
        word_emb = self.word_emb(word_ids)  # (batch_size, w_seq_len, word_dim)
        char_emb = self.char_emb(char_ids)  # (batch_size, w_seq_len, 100)
        # emb = torch.cat([word_emb, char_emb], dim=2)  # (batch_size, w_seq_len, word_dim + 100)
        emb = self.linear(word_emb)  # (batch_size, w_seq_len, dim)
        # if self.training:
        #     _zero_idxs = torch.zeros(emb.shape[:2], device=emb.device, dtype=torch.long)
        #     emb[word_masks] = self.mask_embedding(_zero_idxs)[word_masks].to(emb.dtype)  # make AMP happy
        return emb

class PositionalEmbedding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, num_embeddings, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, inputs):
        bsz, seq_length = inputs.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class VisualProjection(nn.Module):
    def __init__(self, visual_dim, dim, drop_rate=0.0):
        super(VisualProjection, self).__init__()
        self.drop = nn.Dropout(p=drop_rate)
        self.linear = Conv1D(in_dim=visual_dim, out_dim=dim, kernel_size=1, stride=1, bias=True, padding=0)

    def forward(self, visual_features):
        # the input visual feature with shape (batch_size, seq_len, visual_dim)
        visual_features = self.drop(visual_features)
        output = self.linear(visual_features)  # (batch_size, seq_len, dim)
        return output


class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, dim, kernel_size, drop_rate, num_layers=4):
        super(DepthwiseSeparableConvBlock, self).__init__()
        self.depthwise_separable_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, groups=dim,
                          padding=kernel_size // 2, bias=False),
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, bias=True),
                nn.ReLU(),
            ) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim, eps=1e-6) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        output = x  # (batch_size, seq_len, dim)
        for idx, conv_layer in enumerate(self.depthwise_separable_conv):
            residual = output
            output = self.layer_norms[idx](output)  # (batch_size, seq_len, dim)
            output = output.transpose(1, 2)  # (batch_size, dim, seq_len)
            output = conv_layer(output)
            output = self.dropout(output)
            output = output.transpose(1, 2) + residual  # (batch_size, seq_len, dim)
        return output

class BiLinear(nn.Module):
    def __init__(self, dim):
        super(BiLinear, self).__init__()
        self.dense_1 = Conv1D(in_dim=dim, out_dim=dim,
                              kernel_size=1, stride=1, padding=0, bias=True)
        self.dense_2 = Conv1D(in_dim=dim, out_dim=dim,
                              kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, input1, input2):
        output = self.dense_1(input1) + self.dense_2(input2)
        return output

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, drop_rate):
        super(MultiHeadAttentionBlock, self).__init__()
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.key = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.value = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, mask=None):
        output = self.layer_norm1(x)  # (batch_size, seq_len, dim)
        output = self.dropout(output)
        # multi-head attention layer
        query = self.transpose_for_scores(self.query(output))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output))
        value = self.transpose_for_scores(self.value(output))
        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(attention_probs, value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = output + x
        output = self.layer_norm2(residual)
        output = self.dropout(output)
        output = self.out_layer(output)
        output = self.dropout(output) + residual
        return output


class FeatureEncoder(nn.Module):
    def __init__(self, dim, num_heads, max_pos_len, kernel_size=7, num_layers=4, drop_rate=0.0):
        super(FeatureEncoder, self).__init__()
        self.pos_embedding = PositionalEmbedding(num_embeddings=max_pos_len, embedding_dim=dim)
        self.conv_block = DepthwiseSeparableConvBlock(dim=dim, kernel_size=kernel_size, drop_rate=drop_rate,
                                                      num_layers=num_layers)
        self.attention_block = MultiHeadAttentionBlock(dim=dim, num_heads=num_heads, drop_rate=drop_rate)

    def forward(self, x, mask=None):
        features = x + self.pos_embedding(x)  # (batch_size, seq_len, dim)
        features = self.conv_block(x)  # (batch_size, seq_len, dim)
        features = self.attention_block(features, mask=mask)  # (batch_size, seq_len, dim)
        return features


class SGPAEncoder(nn.Module):
    def __init__(self, dim, 
                    max_snippet_len, 
                    num_heads = 4, 
                    num_layers = 4, 
                    drop_rate = 0.1,
                    shared = False, 
                    use_contrastive = False, 
                    **kwargs):
        """
        Self-Guided Parallel Attention Module
        """
        super(SGPAEncoder, self).__init__()
        self.shared = shared
        self.use_contrastive = use_contrastive
        self._create_sgpa_layers(dim, num_heads, num_layers, drop_rate)
        
        # learnable token
        if self.use_contrastive:
            self.emb_token = nn.Parameter(torch.randn(dim))
    

    def _create_sgpa_layers(self, dim, num_heads, num_layers, drop_rate):
        sgpa = DualMultiHeadAttentionConvBlock(dim, num_heads, drop_rate)
        if self.shared:
            self.sgpa_layers=nn.ModuleList([copy.deepcopy(sgpa) for i in range(num_layers)])
        else:
            self.sgpa_m_layers=nn.ModuleList([copy.deepcopy(sgpa) for i in range(num_layers)])
            self.sgpa_q_layers=nn.ModuleList([copy.deepcopy(sgpa) for i in range(num_layers)])


    def forward(self, mfeats, qfeats, m_mask, q_mask, h_mask=None, *args):
        assert mfeats.shape[2] == 1, 'SGPA Encoder does not support spatial interaction!'
        bs, seq_len = mfeats.shape[:2]
        mfeats = mfeats.view(bs, seq_len, -1)
        hfeats = None
        pure_qfeats = qfeats
        if self.use_contrastive:
            # Each batch has its own set of tokens
            emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)
            # adding the embedding token for all sequences
            hfeats = torch.cat((emb_token[:, None], mfeats), 1)

            # create a bigger mask, to allow attend to emb
            token_mask = torch.ones((bs, 1), dtype=bool, device=mfeats.device)
            h_mask = torch.cat((token_mask, h_mask), 1)
            # hfea  ts = self.pos_embed(hfeats)

        # w/o sometimes better
        # mfeats = self.pos_embed(mfeats)
        # qfeats = self.pos_embed(qfeats)

        if self.shared:
            sgpa_layers = self.sgpa_layers
        else:
            sgpa_layers = zip(self.sgpa_m_layers, self.sgpa_q_layers)

        for sgpa_layer in sgpa_layers:
            if self.shared:
                # Two modalities share one encoder to obtain more robust features
                sgpa_m = sgpa_layer
                mfeats_ = sgpa_m(from_tensor=mfeats, to_tensor=qfeats, from_mask=m_mask, to_mask=q_mask)
                qfeats_ = sgpa_m(from_tensor=qfeats, to_tensor=mfeats, from_mask=q_mask, to_mask=m_mask)
            else:
                sgpa_m, sgpa_q = sgpa_layer
                mfeats_ = sgpa_m(from_tensor=mfeats, to_tensor=qfeats, from_mask=m_mask,to_mask=q_mask)
                qfeats_ = sgpa_q(from_tensor=qfeats, to_tensor=mfeats, from_mask=q_mask, to_mask=m_mask)\

            # Input the masked features into the SGPA
            if self.use_contrastive:
                hfeats = sgpa_m(hfeats, hfeats, h_mask, h_mask)
                if self.shared:
                    pure_qfeats = sgpa_m(pure_qfeats, pure_qfeats, q_mask, q_mask)
                else:
                    pure_qfeats = sgpa_q(pure_qfeats, pure_qfeats, q_mask, q_mask)
        
            mfeats, qfeats = mfeats_, qfeats_
            
        if self.use_contrastive:
            # Extract the tokens of the corresponding position from the output features
            query_emb = pure_qfeats[:,0,:]
            motion_emb = hfeats[:,0,:]

            return mfeats, qfeats, query_emb, motion_emb

        return mfeats, qfeats, None, None

def create_attention_mask(from_mask, to_mask, broadcast_ones=False):
    batch_size, from_seq_len = from_mask.size()
    _, to_seq_len = to_mask.size()
    to_mask = to_mask.unsqueeze(1).float()

    if broadcast_ones:
        mask = torch.ones(batch_size, from_seq_len, 1).float()
    else:
        mask = from_mask.unsqueeze(2).float()

    mask = torch.matmul(mask, to_mask)  # (batch_size, from_seq_len, to_seq_len)
    return mask


class DualMultiHeadAttentionConvBlock(nn.Module):
    def __init__(self, dim, num_heads, drop_rate):
        """
        Self-Guided Parallel Attention Module Pytorch Implementation
        """
        super(DualMultiHeadAttentionConvBlock, self).__init__()
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)

        self.query = Conv1D(in_dim=dim, out_dim=dim)
        self.f_key = Conv1D(in_dim=dim, out_dim=dim)
        self.f_value = Conv1D(in_dim=dim, out_dim=dim)
        self.s_proj = Conv1D(in_dim=dim, out_dim=dim)
        self.t_key = Conv1D(in_dim=dim, out_dim=dim)
        self.t_value = Conv1D(in_dim=dim, out_dim=dim)
        self.x_proj = Conv1D(in_dim=dim, out_dim=dim)
        self.s_gate = Conv1D(in_dim=dim, out_dim=dim)
        self.x_gate = Conv1D(in_dim=dim, out_dim=dim)
        self.bilinear_1 = BiLinear(dim)
        self.bilinear_2 = BiLinear(dim)
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_normt = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        self.guided_dense = Conv1D(in_dim=dim, out_dim=dim)
        self.output_dense = Conv1D(in_dim=dim, out_dim=dim)
        self.out_layer = Conv1D(in_dim=dim, out_dim=dim)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        # (batch_size, num_heads, w_seq_len, head_size)
        return x.permute(0, 2, 1, 3)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)
    
    def compute_attention(self, query, key, value, mask):
        attn_value = torch.matmul(query, key.transpose(-1, -2))
        attn_value = attn_value / math.sqrt(self.head_size)
        attn_value += (1.0 - mask) * -1e30
        attn_score = nn.Softmax(dim=-1)(attn_value)
        attn_score = self.dropout(attn_score)
        out = torch.matmul(attn_score, value)
        return out

    def forward(self, from_tensor, to_tensor, from_mask, to_mask):
        x = from_tensor
        from_tensor = self.layer_norm1(from_tensor)  # (batch_size, from_seq_len, dim)
        to_tensor = self.layer_normt(to_tensor)  # (batch_size, to_seq_len, dim)
        # dual multi-head attention layer
        # self-attn projection (batch_size, num_heads, from_seq_len, head_size)
        query = self.transpose_for_scores(self.query(from_tensor))
        f_key = self.transpose_for_scores(self.f_key(from_tensor))
        f_value = self.transpose_for_scores(self.f_value(from_tensor))
        # cross-attn projection (batch_size, num_heads, to_seq_len, head_size)
        t_key = self.transpose_for_scores(self.t_key(to_tensor))
        t_value = self.transpose_for_scores(self.t_key(to_tensor))
        # create attention mask
        s_attn_mask = create_attention_mask(from_mask, from_mask, broadcast_ones=False).unsqueeze(1)
        x_attn_mask = create_attention_mask(from_mask, to_mask, broadcast_ones=False).unsqueeze(1)
        # compute self-attention
        s_value = self.compute_attention(query, f_key, f_value, s_attn_mask)
        s_value = self.combine_last_two_dim(s_value.permute(0, 2, 1, 3))  # (batch_size, from_seq_len, dim)
        s_value = self.s_proj(s_value)
        # compute cross-attention
        x_value = self.compute_attention(query, t_key, t_value, x_attn_mask)
        x_value = self.combine_last_two_dim(x_value.permute(0, 2, 1, 3))  # (batch_size, from_seq_len, dim)
        x_value = self.x_proj(x_value)
        # cross gating strategy
        s_score = nn.Sigmoid()(self.s_gate(s_value))
        x_score = nn.Sigmoid()(self.x_gate(x_value))
        outputs = s_score * x_value + x_score * s_value
        outputs = self.guided_dense(outputs)
        # self-guided
        scores = self.bilinear_1(from_tensor, outputs)
        values = self.bilinear_2(from_tensor, outputs)
        output = nn.Sigmoid()(mask_logits(scores, from_mask.unsqueeze(2))) * values
        outputs = self.output_dense(output)
        # intermediate layer
        output = self.dropout(output)
        residual = output + x
        output = self.layer_norm2(residual)
        output = self.dropout(output)
        output = self.out_layer(output)
        output = self.dropout(output) + residual
        return output

class CQAttention(nn.Module):
    def __init__(self, dim, drop_rate=0.0):
        super(CQAttention, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, c_mask, q_mask):
        score = self.trilinear_attention(context, query)  # (batch_size, c_seq_len, q_seq_len)
        score_ = nn.Softmax(dim=2)(mask_logits(score, q_mask.unsqueeze(1)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = score_t.transpose(1, 2)  # (batch_size, q_seq_len, c_seq_len)
        c2q = torch.matmul(score_, query)  # (batch_size, c_seq_len, dim)
        q2c = torch.matmul(torch.matmul(score_, score_t), context)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, c2q, torch.mul(context, c2q), torch.mul(context, q2c)], dim=2)
        output = self.cqa_linear(output)  # (batch_size, c_seq_len, dim)
        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand([-1, -1, q_seq_len])  # (batch_size, c_seq_len, q_seq_len)
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand([-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)
        return res


class WeightedPool(nn.Module):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        alpha = torch.tensordot(x, self.weight, dims=1)  # shape = (batch_size, seq_length, 1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(1, 2), alphas)  # (batch_size, dim, 1)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x


class CQConcatenate(nn.Module):
    def __init__(self, dim):
        super(CQConcatenate, self).__init__()
        self.weighted_pool = WeightedPool(dim=dim)
        self.conv1d = Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, q_mask):
        pooled_query = self.weighted_pool(query, q_mask)  # (batch_size, dim)
        _, c_seq_len, _ = context.shape
        pooled_query = pooled_query.unsqueeze(1).repeat(1, c_seq_len, 1)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, pooled_query], dim=2)  # (batch_size, c_seq_len, 2*dim)
        output = self.conv1d(output)
        return output


class HighLightLayer(nn.Module):
    def __init__(self, dim, drop_rate, max_snippet_len):
        super(HighLightLayer, self).__init__()
        self.conv1d = Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, mask):
        # compute logits
        logits = self.conv1d(x)
        logits = logits.squeeze(2)
        logits = mask_logits(logits, mask)
        # compute score
        scores = nn.Sigmoid()(logits)
        return scores

    @staticmethod
    def compute_loss(scores, labels, mask, epsilon=1e-12):
        labels = labels.type(torch.float32)
        weights = torch.where(labels == 0.0, labels + 1.0, 2.0 * labels)
        loss_per_location = nn.BCELoss(reduction='none')(scores, labels)
        loss_per_location = loss_per_location * weights
        mask = mask.type(torch.float32)
        loss = torch.sum(loss_per_location * mask) / (torch.sum(mask) + epsilon)
        return loss

class NERHighLightLayer(nn.Module):
    def __init__(self, dim, drop_rate, max_snippet_len):
        super(NERHighLightLayer, self).__init__()
        self.conv1d = Conv1D(in_dim=dim, out_dim=4, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, mask):
        # compute logits
        logits = self.conv1d(x)
        # logits = logits.squeeze(2)
        logits = mask_logits(logits, mask.unsqueeze(-1))
        # compute score
        # scores = nn.Sigmoid()(logits)
        return logits

    @staticmethod
    def compute_loss(m_logits, m_labels, label_embs, vmask, epsilon=1e-12):
        log_probs = F.log_softmax(m_logits, dim=-1)
        # NLLLoss
        # loss_fun = nn.NLLLoss()
        # loss_fun = nn.CrossEntropyLoss()
        m_labels = F.one_hot(m_labels).float()
        # m_loss = loss_fun(m_probs, m_labels)
        # m_loss = loss_fun(m_probs.transpose(1,2), m_labels)

        loss_per_sample = -torch.sum(m_labels * log_probs, dim=-1)
        # m_loss =torch.sum(loss_per_sample * vmask, dim=-1) / (torch.sum(vmask, dim=-1) + 1e-12)
        # m_loss = m_loss.mean()
        m_loss =torch.sum(loss_per_sample * vmask) / (torch.sum(vmask) + epsilon)
        
        # add punishment
        ortho_constraint = torch.matmul(label_embs.T, label_embs) * (1.0 - torch.eye(4, device=label_embs.device, dtype=torch.float32))
        ortho_constraint = torch.norm(ortho_constraint, p=2)  # compute l2 norm as loss
        m_loss += ortho_constraint
        return m_loss

class DynamicRNN(nn.Module):
    def __init__(self, dim):
        super(DynamicRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, bias=True, batch_first=True,
                            bidirectional=False)

    def forward(self, x, mask):
        out, _ = self.lstm(x)  # (bsz, seq_len, dim)
        mask = mask.type(torch.float32)
        mask = mask.unsqueeze(2)
        out = out * mask
        return out


class ConditionedPredictor(nn.Module):
    def __init__(self, dim, num_heads, max_pos_len, drop_rate=0.0, predictor='rnn'):
        super(ConditionedPredictor, self).__init__()
        self.predictor = predictor
        if predictor == 'rnn':
            self.start_encoder = DynamicRNN(dim=dim)
            self.end_encoder = DynamicRNN(dim=dim)
        else:
            self.encoder = FeatureEncoder(dim=dim, num_heads=num_heads, kernel_size=7, num_layers=4,
                                          max_pos_len=max_pos_len, drop_rate=drop_rate)
            self.start_layer_norm = nn.LayerNorm(dim, eps=1e-6)
            self.end_layer_norm = nn.LayerNorm(dim, eps=1e-6)

        self.start_block = nn.Sequential(
            Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.end_block = nn.Sequential(
            Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x, mask):
        if self.predictor == 'rnn':
            start_features = self.start_encoder(x, mask)  # (batch_size, seq_len, dim)
            end_features = self.end_encoder(start_features, mask)
        else:
            start_features = self.encoder(x, mask)
            end_features = self.encoder(start_features, mask)
            start_features = self.start_layer_norm(start_features)
            end_features = self.end_layer_norm(end_features)
        start_features = self.start_block(torch.cat([start_features, x], dim=2))  # (batch_size, seq_len, 1)
        end_features = self.end_block(torch.cat([end_features, x], dim=2))
        start_logits = mask_logits(start_features.squeeze(2), mask=mask)
        end_logits = mask_logits(end_features.squeeze(2), mask=mask)
        return start_logits, end_logits

    @staticmethod
    def extract_index(start_logits, end_logits):
        start_prob = nn.Softmax(dim=1)(start_logits)
        end_prob = nn.Softmax(dim=1)(end_logits)
        outer = torch.matmul(start_prob.unsqueeze(dim=2), end_prob.unsqueeze(dim=1))
        outer = torch.triu(outer, diagonal=0)
        _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)  # (batch_size, )
        _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)  # (batch_size, )
        return start_index, end_index

    @staticmethod
    def compute_cross_entropy_loss(start_logits, end_logits, start_labels, end_labels):
        start_loss = nn.CrossEntropyLoss(reduction='mean')(start_logits, start_labels)
        end_loss = nn.CrossEntropyLoss(reduction='mean')(end_logits, end_labels)
        return start_loss + end_loss

class DynamicRNN2(nn.Module):
    def __init__(self, dim):
        super(DynamicRNN2, self).__init__()
        self.rnn = nn.GRU(input_size=dim, hidden_size=dim, num_layers=1, bias=True, batch_first=True,
                            bidirectional=False)

    def forward(self, x, mask):
        out, _ = self.rnn(x)  # (bsz, seq_len, dim)
        mask = mask.type(torch.float32)
        mask = mask.unsqueeze(2)
        out = out * mask
        return out


class LabelPriorPredictor(nn.Module):
    def __init__(self, dim, drop_rate=0.0, mask_ratio=0.5):
        super(LabelPriorPredictor, self).__init__()
        self.label_noise_prob = mask_ratio
        self.start_encoders = DynamicRNN2(dim)
        self.end_encoders = DynamicRNN2(dim)
        self.t = 1.0

        # start end mid label
        self.num_classes = 2
        self.label_encoder = nn.Embedding(self.num_classes, dim)
        
        self.start_block = nn.Sequential(
            Conv1D(in_dim=dim, out_dim=dim, kernel_size=1,
                   stride=1, padding=0, bias=True),
            nn.LayerNorm(dim, eps=1e-6),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1,
                   stride=1, padding=0, bias=True),
        )
        self.end_block = nn.Sequential(
            Conv1D(in_dim=dim, out_dim=dim, kernel_size=1,
                   stride=1, padding=0, bias=True),
            nn.LayerNorm(dim, eps=1e-6),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1,
                   stride=1, padding=0, bias=True),
        )

    def forward(self, x, mask, s_labels, e_labels):
        if not self.training:
            start_features = x
            start_features = self.start_encoders(start_features, mask)
            end_features = self.end_encoders(start_features, mask)
            start_features = self.start_block(start_features)  # (batch_size, seq_len, 1)
            end_features = self.end_block(end_features)
            start_logits = mask_logits(start_features.squeeze(2), mask=mask)
            end_logits = mask_logits(end_features.squeeze(2), mask=mask)
            return start_logits, end_logits, None, None
        batch_size, len, dim = x.shape
        nfeats = mask.sum(1).to(torch.int)
        # org_s_label_queries = self.s_label_encoder(s_labels)
        # org_e_label_queries = self.e_label_encoder(e_labels)
        # org_label_queries = self.label_encoder(torch.clamp(s_labels+e_labels, max=1))
        s_new_labels = torch.zeros_like(mask)
        e_new_labels = torch.zeros_like(mask)
        row_idx = torch.arange(0, batch_size).long()
        s_new_labels[row_idx, s_labels] = 1
        e_new_labels[row_idx, e_labels] = 1
        s_new_labels = s_new_labels.flatten()
        e_new_labels = e_new_labels.flatten()
        # perturb labels
        noise_prob = self.label_noise_prob
        # noised_s_labels = apply_label_noise(s_labels, h_labels, noise_prob, mask)
        # noised_e_labels = apply_label_noise(e_labels, h_labels, noise_prob, mask)
        # noised_s_labels = apply_label_noise(s_labels, noise_prob, nfeats, self.PAD_LABEL_ID)
        # noised_e_labels = apply_label_noise(e_labels, noise_prob, nfeats, self.PAD_LABEL_ID)   
        noised_s_labels = apply_label_noise(s_new_labels.long(), noise_prob, self.num_classes)
        noised_e_labels = apply_label_noise(e_new_labels.long(), noise_prob, self.num_classes)
        noised_s_labels = noised_s_labels.reshape(batch_size,  -1)
        noised_e_labels = noised_e_labels.reshape(batch_size,  -1)
        
        mask = mask.repeat(1, 2)

        # (batch_size, seq_len, dim)
        # x = self.scale(x)
        
        start_features = x
        # commit_features = start_features
        # # encoding labels
        noised_s_label_queries = self.label_encoder(noised_s_labels)
        noised_e_label_queries = self.label_encoder(noised_e_labels)
        start_features = torch.cat((start_features, noised_s_label_queries), dim=1)
        start_features = self.start_encoders(start_features, mask)
        end_features = torch.cat((start_features[:, :len], noised_e_label_queries), dim=1)
        end_features = self.end_encoders(end_features, mask)

        start_features = self.start_block(start_features)  # (batch_size, seq_len, 1)
        end_features = self.end_block(end_features)
        start_logits = mask_logits(start_features.squeeze(2), mask=mask)
        end_logits = mask_logits(end_features.squeeze(2), mask=mask)

        dn_s_logits = start_logits[:, len:].contiguous()
        dn_e_logits = end_logits[:, len:].contiguous()
        start_logits = start_logits[:, :len].contiguous()
        end_logits = end_logits[:, :len].contiguous()
        
        # self._set_aux_loss(all_dn_s_logits, all_dn_e_logits, outputs)
        return start_logits, end_logits, dn_s_logits, dn_e_logits

    @torch.jit.unused
    def _set_aux_loss(self, all_dn_s_logits, all_dn_e_logits, outputs):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        for i, (a, b) in enumerate(zip(all_dn_s_logits, all_dn_e_logits)):
            outputs.update({f"dn_start_loc_{i}": a})
            outputs.update({f"dn_end_loc_{i}": b})
            outputs.update({f"dn_start_kl_loc_{i}": a})
            outputs.update({f"dn_end_kl_loc_{i}": b})
    
    @staticmethod
    def compute_cross_entropy_loss(start_logits, end_logits, start_labels, end_labels):
        start_loss = nn.CrossEntropyLoss(reduction='mean')(start_logits, start_labels)
        end_loss = nn.CrossEntropyLoss(reduction='mean')(end_logits, end_labels)
        return start_loss + end_loss
    
    @staticmethod
    def compute_kl_loss(start_logits, end_logits, dn_s_logits, dn_e_logits, mask):
        t = 1.0
        masked_start_logits = start_logits * mask
        masked_end_logits = end_logits * mask
        masked_dn_s_logits = dn_s_logits * mask
        masked_dn_e_logits = dn_e_logits * mask

        s_prob = F.log_softmax(masked_start_logits / t, dim=-1)
        dn_s_prob = F.softmax(masked_dn_s_logits / t, dim=-1)
        start_loss = F.kl_div(s_prob, dn_s_prob, reduction='batchmean')
        
        e_prob = F.log_softmax(masked_end_logits / t, dim=-1)
        dn_e_prob = F.softmax(masked_dn_e_logits / t, dim=-1)
        end_loss = F.kl_div(e_prob, dn_e_prob, reduction='batchmean')
        
        return start_loss + end_loss
    
    @staticmethod
    def extract_index(start_logits, end_logits):
        start_prob = nn.Softmax(dim=1)(start_logits)
        end_prob = nn.Softmax(dim=1)(end_logits)
        outer = torch.matmul(start_prob.unsqueeze(dim=2), end_prob.unsqueeze(dim=1))
        outer = torch.triu(outer, diagonal=0)
        _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)  # (batch_size, )
        _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)  # (batch_size, )
        return start_index, end_index


def apply_label_noise(
    labels: torch.Tensor,
    label_noise_prob: float = 0.2,
    num_classes: int = 80,
):
    """
    Args:
        labels (torch.Tensor): Classification labels with ``(num_labels, )``.
        label_noise_prob (float): The probability of the label being noised. Default: 0.2.
        num_classes (int): Number of total categories.

    Returns:
        torch.Tensor: The noised labels the same shape as ``labels``.
    """
    if label_noise_prob > 0:
        p = torch.rand_like(labels.float())
        noised_index = torch.nonzero(p < label_noise_prob).view(-1)
        new_lebels = torch.randint_like(noised_index, 0, num_classes)
        noised_labels = labels.scatter_(0, noised_index, new_lebels)
        return noised_labels
    else:
        return labels