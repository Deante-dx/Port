import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers_t7 import Embedding, VisualProjection, FeatureEncoder, CQAttention, CQConcatenate, \
    ConditionedPredictor, HighLightLayer, NERHighLightLayer, SGPAEncoder, LabelPriorPredictor
from transformers import AdamW, get_linear_schedule_with_warmup


def build_optimizer_and_scheduler(model, configs):
    no_decay = ['bias', 'layer_norm', 'LayerNorm']  # no decay for parameters of layer norm and bias
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.init_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, configs.num_train_steps * configs.warmup_proportion,
                                                configs.num_train_steps)
    return optimizer, scheduler


class VSLNet(nn.Module):
    def __init__(self, configs, word_vectors):
        super(VSLNet, self).__init__()
        self.configs = configs
        self.embedding_net = Embedding(num_words=configs.word_size, num_chars=configs.char_size, out_dim=configs.dim,
                                       word_dim=configs.word_dim, char_dim=configs.char_dim, word_vectors=word_vectors,
                                       drop_rate=configs.drop_rate)
        # self.embedding_net = BERTEncoder(modelpath='deps/distilbert-base-uncased', finetune=True, latent_dim=configs.dim, drop_rate=configs.drop_rate)
        self.video_affine = VisualProjection(visual_dim=configs.video_feature_dim, dim=configs.dim,
                                             drop_rate=configs.drop_rate)
        self.feature_encoder = SGPAEncoder(dim=configs.dim, max_snippet_len=configs.max_pos_len, num_heads=configs.num_heads, drop_rate=configs.drop_rate, num_layers=configs.sgpa_layers, shared=True)
        # FeatureEncoder(dim=configs.dim, num_heads=configs.num_heads, kernel_size=7, num_layers=4,
        #                                       max_pos_len=configs.max_pos_len, drop_rate=configs.drop_rate)
        # video and query fusion
        self.cq_attention = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        self.cq_concat = CQConcatenate(dim=configs.dim)
        # query-guided highlighting
        # self.highlight_layer = NERHighLightLayer(dim=configs.dim, drop_rate=configs.drop_rate, max_snippet_len=128)
        self.highlight_layer = HighLightLayer(dim=configs.dim, drop_rate=configs.drop_rate, max_snippet_len=128)
        # conditioned predictor
        # self.predictor = ConditionedPredictor(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate,
        #                                       max_pos_len=configs.max_pos_len, predictor=configs.predictor)
        self.predictor = LabelPriorPredictor(dim=configs.dim, drop_rate=configs.drop_rate, mask_ratio=configs.beta)
        # self.text_mlp = nn.Linear(configs.dim, configs.word_size)
        self.text_mlp = nn.Sequential(
            nn.Linear(configs.dim, configs.dim//4),
            nn.ReLU(),
            nn.Dropout(configs.drop_rate),
            nn.Linear(configs.dim//4, configs.word_size)
        )
        self.t = nn.Parameter(1.0 * torch.ones([]))

        # lable_emb = torch.empty(size=[configs.dim, 4], dtype=torch.float32)
        # lable_emb = torch.nn.init.orthogonal_(lable_emb.data)
        # self.label_embs = nn.Parameter(lable_emb, requires_grad=True)
        
        # init parameters
        self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()
        self.apply(init_weights)


    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask, s_labels=None, e_labels=None, h_labels=None, word_masks=None, sentences=None):
        B = v_mask.shape[0]
        video_features = self.video_affine(video_features)
        query_features = self.embedding_net(word_ids, char_ids, word_masks)
        # query_features, q_mask = self.embedding_net(sentences)
        video_features, query_features, _, _ = self.feature_encoder(video_features.unsqueeze(2), query_features, v_mask, q_mask)
        text_logits = self.text_mlp(query_features)
        # query_features = self.feature_encoder(, mask=q_mask)
        features = self.cq_attention(video_features, query_features, v_mask, q_mask)
        features = self.cq_concat(features, query_features, q_mask)
        # q2v_features = self.cq_attention(video_features, query_features, v_mask, q_mask)
        # v2q_features = self.cq_attention(query_features, video_features, q_mask, v_mask)
        # features = self.cq_concat(q2v_features, v2q_features, q_mask)
        h_score = self.highlight_layer(features, v_mask)
        features = features * h_score.unsqueeze(2)
        # match_logits = self.highlight_layer(features, v_mask)
        # match_score = F.gumbel_softmax(match_logits, tau=1.0)
        # # match_probs = torch.log(match_score)
        # soft_label_embs = torch.matmul(match_score, torch.tile(self.label_embs, (B, 1, 1)).permute(0, 2, 1))
        # features = (features + soft_label_embs) * v_mask.unsqueeze(2)
        start_logits, end_logits, dn_s_logits, dn_e_logits = self.predictor(features, v_mask, s_labels, e_labels)
        return h_score, start_logits, end_logits, text_logits, dn_s_logits, dn_e_logits, None

    def extract_index(self, start_logits, end_logits):
        return self.predictor.extract_index(start_logits=start_logits, end_logits=end_logits)

    def compute_highlight_loss(self, scores, labels, mask):
        return self.highlight_layer.compute_loss(scores=scores, labels=labels, mask=mask)

    # def compute_NER_highlight_loss(self, match_logits, labels, label_embs, mask):
    #     return self.highlight_layer.compute_loss(match_logits, labels, label_embs, mask)

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor.compute_cross_entropy_loss(start_logits=start_logits, end_logits=end_logits,
                                                         start_labels=start_labels, end_labels=end_labels)
    
    def compute_kl_loss(self, start_logits, end_logits, dn_s_logits, dn_e_logits, mask):
        # return self.predictor.compute_kl_loss(start_logits, end_logits, dn_s_logits, dn_e_logits, mask)
        masked_start_logits = start_logits * mask
        masked_end_logits = end_logits * mask
        masked_dn_s_logits = dn_s_logits * mask
        masked_dn_e_logits = dn_e_logits * mask

        s_prob = F.log_softmax(masked_start_logits / self.t, dim=-1)
        dn_s_prob = F.softmax(masked_dn_s_logits / self.t, dim=-1)
        start_loss = F.kl_div(s_prob, dn_s_prob, reduction='batchmean')
        
        e_prob = F.log_softmax(masked_end_logits / self.t, dim=-1)
        dn_e_prob = F.softmax(masked_dn_e_logits / self.t, dim=-1)
        end_loss = F.kl_div(e_prob, dn_e_prob, reduction='batchmean')

        return start_loss + end_loss
    
    def compute_mask_loss(self, text_logits, word_ids, word_masks):
        mask_loss = F.cross_entropy(text_logits.transpose(1, 2), word_ids, reduction="none") * word_masks.float()
        mask_loss = mask_loss.mean()
        return mask_loss
