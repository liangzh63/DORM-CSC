from tqdm import tqdm
import pickle
from typing import Dict
from transformers import BertTokenizer
from src.utils import Pinyin

def add_pinyin_feature_to_pickle(pickle_path):

    pinyin_bert_path = "models/chinese_roberta_csc_pinyin"
    pinyin_convertor = Pinyin()
    tokenizer = BertTokenizer.from_pretrained(pinyin_bert_path)
    
    data_list = pickle.load(open(pickle_path, "rb"))
    for item in tqdm(data_list):
        src_idx = item["src_idx"]

        sm_ids = []
        ym_ids = []
        sd_ids = []
        ## src_ids 并非与 src 一一对应，所以需要对 src_idx 而不是对 src 提取拼音
        for token_id in src_idx[1:-1]:
            token = tokenizer.convert_ids_to_tokens([token_id])[0]
            if not token.startswith('##') and len(token) == 1:
                sm, ym, sd = pinyin_convertor.get_pinyin(token)
            else:
                sm, ym, sd = "[UNK]", "[UNK]", "[UNK]"    
            sm_ids.append( tokenizer.vocab.get("pinyin_" + str(sm)))
            ym_ids.append( tokenizer.vocab.get("pinyin_" + str(ym)))
            sd_ids.append( tokenizer.vocab.get("pinyin_" + str(sd)))

        assert len(sm_ids) == len(ym_ids)
        assert len(ym_ids) == len(sd_ids)
        assert len(sm_ids) == len(src_idx) - 2 # 减去 [CLS] and [SEP]
        
        item["sm_ids"] = sm_ids
        item["ym_ids"] = ym_ids
        item["sd_ids"] = sd_ids
        
    
    new_pickle_path = pickle_path.replace(".pkl", "_pinyin2.pkl")
    pickle.dump(data_list, open(new_pickle_path, "wb"))
    print(f"旧的pickle文件为{pickle_path}, 已为其加入拼音特征, 已存储在{new_pickle_path}")



if __name__ == "__main__":
    
    path_list = [
        # "data/test.sighan13.pkl", 
        # "data/test.sighan14.pkl",
        # "data/test.sighan15.pkl",
        "data/trainall.times2.pkl"
    ]
    for src_pickle_path in path_list:
        add_pinyin_feature_to_pickle(src_pickle_path)
    


    
    
    
    
    
    
    
    
    