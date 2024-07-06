import numpy as np
import torch
import torch.utils.data
from util.data_util import pad_seq, pad_char_seq, pad_video_seq

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_features, mask_rate=0.15):
        super(Dataset, self).__init__()
        self.dataset = dataset
        self.video_features = video_features
        self.mask_rate = mask_rate

    def __getitem__(self, index):
        record = self.dataset[index]
        video_feature = self.video_features[record['vid']]
        s_ind, e_ind = int(record['s_ind']), int(record['e_ind'])
        word_ids, char_ids = record['w_ids'], record['c_ids']
        word_mask = self.get_word_mask(word_ids)
        sentence = record['sentence']
        return record, video_feature, sentence, word_ids, char_ids, s_ind, e_ind, word_mask

    def __len__(self):
        return len(self.dataset)
    
    def get_word_mask(self, text_inds):
        mask_rate = self.mask_rate

        length_text = len(text_inds)
        word_mask = [np.random.uniform() < mask_rate for _ in range(length_text)]

        if np.sum(word_mask) == 0 or np.sum(word_mask) == length_text:
            random_idx = np.random.choice(np.arange(length_text))
            word_mask[random_idx] = True

        return word_mask
    
def get_NER_label(self, sidx, eidx, v_len):
        max_len = self.max_vlen
        cur_max_len = v_len
        st, et = sidx, eidx
        NER_label = np.zeros([max_len], dtype=torch.int64) 

        ext_len = 1
        new_st_l = max(0, st - ext_len)
        new_st_r = min(st + ext_len, cur_max_len - 1)
        new_et_l = max(0, et - ext_len)
        new_et_r = min(et + ext_len, cur_max_len - 1)
        if new_st_r >= new_et_l:
            new_st_r = max(st, new_et_l - 1)
        NER_label[new_st_l:(new_st_r + 1)] = 1  # add B-M labels
        NER_label[(new_st_r + 1):new_et_l] = 2  # add I-M labels
        NER_label[new_et_l:(new_et_r + 1)] = 3  # add E-M labels

        return NER_label

def train_collate_fn(data):
    records, video_features, sentences, word_ids, char_ids, s_inds, e_inds, word_masks = zip(*data)
    # process word ids
    word_ids, _ = pad_seq(word_ids)
    word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
    word_masks, _ = pad_seq(word_masks)
    word_masks = np.asarray(word_masks, dtype=np.bool8)  # (batch_size, w_seq_len)
    # process char ids
    char_ids, _ = pad_char_seq(char_ids)
    char_ids = np.asarray(char_ids, dtype=np.int32)  # (batch_size, w_seq_len, c_seq_len)
    # process video features
    vfeats, vfeat_lens = pad_video_seq(video_features)
    vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
    # process labels
    max_len = np.max(vfeat_lens)
    batch_size = vfeat_lens.shape[0]
    s_labels = np.asarray(s_inds, dtype=np.int64)
    e_labels = np.asarray(e_inds, dtype=np.int64)
    h_labels = np.zeros(shape=[batch_size, max_len], dtype=np.int32)
    NER_labels = np.zeros(shape=[batch_size, max_len], dtype=np.int32)
    extend = 0.0
    for idx in range(batch_size):
        # common
        st, et = s_inds[idx], e_inds[idx]
        cur_max_len = vfeat_lens[idx]
        extend_len = round(extend * float(et - st + 1))
        if extend_len > 0:
            st_ = max(0, st - extend_len)
            et_ = min(et + extend_len, cur_max_len - 1)
            h_labels[idx][st_:(et_ + 1)] = 1
        else:
            h_labels[idx][st:(et + 1)] = 1
        # NER
        ext_len = 1
        new_st_l = max(0, st - ext_len)
        new_st_r = min(st + ext_len, cur_max_len - 1)
        new_et_l = max(0, et - ext_len)
        new_et_r = min(et + ext_len, cur_max_len - 1)
        if new_st_r >= new_et_l:
            new_st_r = max(st, new_et_l - 1)
        NER_labels[idx][new_st_l:(new_st_r + 1)] = 1  # add B-M labels
        NER_labels[idx][(new_st_r + 1):new_et_l] = 2  # add I-M labels
        NER_labels[idx][new_et_l:(new_et_r + 1)] = 3  # add E-M labels

    # convert to torch tensor
    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)
    word_ids = torch.tensor(word_ids, dtype=torch.int64)
    word_masks = torch.tensor(word_masks, dtype=torch.bool)
    char_ids = torch.tensor(char_ids, dtype=torch.int64)
    s_labels = torch.tensor(s_labels, dtype=torch.int64)
    e_labels = torch.tensor(e_labels, dtype=torch.int64)
    h_labels = torch.tensor(h_labels, dtype=torch.int64)
    NER_labels = torch.tensor(NER_labels, dtype=torch.int64)
    return records, vfeats, vfeat_lens, sentences, word_ids, char_ids, s_labels, e_labels, h_labels, NER_labels, word_masks


def test_collate_fn(data):
    records, video_features, sentences, word_ids, char_ids, *_ = zip(*data)
    # process word ids
    word_ids, _ = pad_seq(word_ids)
    word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
    # process char ids
    char_ids, _ = pad_char_seq(char_ids)
    char_ids = np.asarray(char_ids, dtype=np.int32)  # (batch_size, w_seq_len, c_seq_len)
    # process video features
    vfeats, vfeat_lens = pad_video_seq(video_features)
    vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
    # convert to torch tensor
    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)
    word_ids = torch.tensor(word_ids, dtype=torch.int64)
    char_ids = torch.tensor(char_ids, dtype=torch.int64)
    return records, vfeats, vfeat_lens, sentences, word_ids, char_ids


def get_train_loader(dataset, video_features, configs):
    train_set = Dataset(dataset=dataset, video_features=video_features, mask_rate=configs.mask_rate)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=configs.batch_size, shuffle=True,
                                               collate_fn=train_collate_fn)
    return train_loader


def get_test_loader(dataset, video_features, configs):
    test_set = Dataset(dataset=dataset, video_features=video_features, mask_rate=configs.mask_rate)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=configs.batch_size, shuffle=False,
                                              collate_fn=test_collate_fn)
    return test_loader
