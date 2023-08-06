from __future__ import absolute_import, division, print_function
import sys
import argparse
import glob
import logging
import os
import random
import json

import numpy as np
import torch


from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)

from transformers import AdamW, get_linear_schedule_with_warmup
from metric import Metric
from dorm import DORM

import pickle

logger = logging.getLogger(__name__)


ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, )), ())

MODEL_CLASSES = {
    "dorm": (BertConfig, DORM, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_dataset(args, input_file):
    input_file = os.path.join(args.data_dir, input_file)
    dataset = pickle.load(open(input_file, 'rb'))
    return dataset


def make_features(args, examples, tokenizer, batch_processor, is_eval=False):
    
    max_length = -1
    for item in examples:
        assert len(item["src_idx"]) == len(item["tgt_idx"])
        max_length = max(max_length, len(item["src_idx"]) + len(item["sm_ids"]) + 1)
    if max_length > args.max_seq_length:
        print(f"参数限制最大长度为{args.max_seq_length}, 当前数据长度为{max_length}")
        assert False
    max_length = min(max_length, args.max_seq_length)
    
    batch = {}
    for t in ['id', 'src', 'tgt', 'tokens_size', 'lengths', 'src_idx', 'tgt_idx', 'masks', 'loss_masks', 
                "pinyin_masks", "pinyin_position_ids", "sm_ids", "ym_ids", "sd_ids"]:
        batch[t] = []

    if args.add_pinyin_loss:
        batch["pinyin_loss_masks"] = []
        batch["pinyin_src_idx"] = []
        batch["pinyin_tgt_idx"] = []

    text_length = (max_length + 1) // 2
    for item in examples:

        for t in item:
            if t == "src_idx" or t == "tgt_idx":
                seq = item[t]  
                padding_length = max_length - len(seq)
                padding_value = 0
                batch[t].append(seq + [padding_value] * padding_length)
                if t == "src_idx":
                    add_pinyin_length = (len(item["sm_ids"]) + 1)
                    text_padding_length = text_length - len(seq)
                    pinyin_padding_length = max_length - text_length - add_pinyin_length
                    batch["masks"].append(               [1] * len(seq) + [0] * text_padding_length + [1] * add_pinyin_length + [0] * pinyin_padding_length   )
                    batch["pinyin_masks"].append(        [0] * text_length + [1] * add_pinyin_length + [0] * pinyin_padding_length )
                    batch["pinyin_position_ids"].append( [0] * text_length + list(range(1, add_pinyin_length+1))       + [0] * pinyin_padding_length )
                    batch["sm_ids"].append(              [0] * text_length + item["sm_ids"] + [tokenizer.sep_token_id] + [0] * pinyin_padding_length )
                    batch["ym_ids"].append(              [0] * text_length + item["ym_ids"] + [tokenizer.sep_token_id] + [0] * pinyin_padding_length )
                    batch["sd_ids"].append(              [0] * text_length + item["sd_ids"] + [tokenizer.sep_token_id] + [0] * pinyin_padding_length )
                    batch["loss_masks"].append(          [0] + [1] * (len(seq)-2) + [0] + [0] * (max_length - len(seq)) )

                    if args.add_pinyin_loss:
                        batch["pinyin_loss_masks"].append( [0] * text_length + [1] * ( len(seq)-2 )  + [0] + [0] * pinyin_padding_length)  # 只算拼音位置的loss
                        batch["pinyin_src_idx"].append(    [0] * text_length + item["src_idx"][1:-1] + [0] + [0] * pinyin_padding_length)  # 只算拼音位置对应的 src_idx
                        batch["pinyin_tgt_idx"].append(    [0] * text_length + item["tgt_idx"][1:-1] + [0] + [0] * pinyin_padding_length)  # 只算拼音位置的 target_idx
                    
                    
            elif t in ["id", "tokens_size", "lengths", "src", "tgt"]:
                batch[t].append(item[t])
        
    

    batch['src_idx'] = torch.tensor(batch['src_idx'], dtype=torch.long)
    batch['tgt_idx'] = torch.tensor(batch['tgt_idx'], dtype=torch.long)
    batch['masks'] = torch.tensor(batch['masks'], dtype=torch.long)
    batch['loss_masks'] = torch.tensor(batch['loss_masks'], dtype=torch.long)
    batch["pinyin_masks"] = torch.tensor(batch["pinyin_masks"], dtype=torch.long)
    batch["pinyin_position_ids"] = torch.tensor(batch["pinyin_position_ids"], dtype=torch.long)
    batch["sm_ids"] = torch.tensor(batch["sm_ids"], dtype=torch.long)
    batch["ym_ids"] = torch.tensor(batch["ym_ids"], dtype=torch.long)
    batch["sd_ids"] = torch.tensor(batch["sd_ids"], dtype=torch.long)
    
    if args.add_pinyin_loss:
        batch["pinyin_loss_masks"] = torch.tensor(batch["pinyin_loss_masks"], dtype=torch.long)
        batch["pinyin_src_idx"] = torch.tensor(batch["pinyin_src_idx"], dtype=torch.long)
        batch["pinyin_tgt_idx"] = torch.tensor(batch["pinyin_tgt_idx"], dtype=torch.long)

    assert batch["src_idx"].size() == batch["tgt_idx"].size()
    assert batch["src_idx"].size() == batch["masks"].size()
    assert batch["src_idx"].size() == batch["loss_masks"].size()
    assert batch["src_idx"].size() == batch["pinyin_masks"].size()
    assert batch["src_idx"].size() == batch["pinyin_position_ids"].size()
    assert batch["src_idx"].size() == batch["sm_ids"].size()
    assert batch["src_idx"].size() == batch["ym_ids"].size()
    assert batch["src_idx"].size() == batch["sd_ids"].size()

    if args.add_pinyin_loss:
        assert batch["src_idx"].size() == batch["pinyin_loss_masks"].size()
        assert batch["src_idx"].size() == batch["pinyin_src_idx"].size()
        assert batch["src_idx"].size() == batch["pinyin_tgt_idx"].size()
    
    batch = batch_processor(batch, tokenizer)
    return batch

def data_helper(args, dataset, tokenizer, batch_processor, is_eval=False):
    if not is_eval:
        random.shuffle(dataset)
        start_position = 0
        width = args.train_batch_size*5000
        intervals = []
        while start_position < len(dataset):
            intervals.append((start_position, min(start_position+width, len(dataset))))
            start_position += width
        bs = args.train_batch_size
    else:
        intervals = [(0, len(dataset))]
        bs = args.eval_batch_size

    for l, r in intervals:
        batches = []
        for i in range(l, r, bs):
            batches.append(make_features(args, dataset[i:min(i+bs,r)], tokenizer, batch_processor))
        for batch in batches:
            yield batch

def train(args, model, tokenizer, batch_processor):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size
    if args.local_rank == -1:
        train_dataset = create_dataset(args, args.train_file)
    else:
        total_dataset = create_dataset(args, args.train_file)
        
        filter_dataset = []
        for item in total_dataset:
            if len(item["src"]) < 140:
                filter_dataset.append(item)
        
        total_dataset = filter_dataset
        start_position = 0
        width = torch.distributed.get_world_size()
        train_dataset = []
        while start_position + width <= len(total_dataset):
            train_dataset.append(total_dataset[start_position+args.local_rank])
            start_position += width 

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataset) // args.train_batch_size // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataset) // args.train_batch_size // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    need_optimized_parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in need_optimized_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in need_optimized_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
            amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset) * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_pinyin_loss, logging_pinyin_loss = 0., 0.
    tr_kl_loss, logging_kl_loss = 0., 0.
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        #epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(data_helper(args, train_dataset, tokenizer, batch_processor, False)):
            model.train()
            for t in batch:
                if t not in ['id', 'src', 'tgt', 'lengths', 'tokens_size', 'pho_lens']:
                    batch[t] = batch[t].to(args.device)
            model_output = model(batch)
            loss = model_output[0]
            pinyin_loss, kl_loss = model_output[-2:]
            if pinyin_loss is None:
                pinyin_loss = torch.zeros(loss.size(), dtype=loss.dtype, device=loss.device)
            if kl_loss is None:
                kl_loss = torch.zeros(loss.size(), dtype=loss.dtype, device=loss.device)
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                pinyin_loss = pinyin_loss / args.gradient_accumulation_steps
                kl_loss = kl_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            tr_pinyin_loss += pinyin_loss.item()
            tr_kl_loss += kl_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss
                    
                    kl_loss_scalar = (tr_kl_loss - logging_kl_loss) / args.logging_steps
                    logs["kl_loss"] = kl_loss_scalar
                    logging_kl_loss = tr_kl_loss
                    
                    pinyin_loss_scalar = (tr_pinyin_loss - logging_pinyin_loss) / args.logging_steps
                    logs["pinyin_loss"] = pinyin_loss_scalar
                    logging_pinyin_loss = tr_pinyin_loss
                    
                    logger.info("Step: {}, LR: {}, Loss: {}, py_loss: {}, kl_loss: {}".format(global_step, logs['learning_rate'], logs['loss'], logs["pinyin_loss"], logs["kl_loss"]))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, 'saved_ckpt-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, batch_processor, prefix=""):
    eval_dataset = create_dataset(args, args.dev_file)
    args.eval_batch_size = args.per_gpu_eval_batch_size 

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0

    batches = []

    for batch in data_helper(args, eval_dataset, tokenizer, batch_processor, True):
        model.eval()
        for t in batch:
            if t not in ['id', 'src', 'tgt', 'lengths', 'tokens_size', 'pho_lens']:
                batch[t] = batch[t].to(args.device)
        with torch.no_grad():
            outputs = model(batch)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=-1)
        batch['src_idx'] = batch['src_idx'].detach().cpu().numpy()
        batch['pred_idx'] = preds

        batches.append(batch)

    metric = Metric(vocab_path=args.output_dir)
    comment = args.comment if hasattr(args, "comment") else ""
    pred_txt_path = os.path.join(args.output_dir, prefix, comment + "preds.txt")
    pred_lbl_path = os.path.join(args.output_dir, prefix, comment + "labels.txt")
    results = metric.metric(
            batches=batches,
            pred_txt_path=pred_txt_path,
            pred_lbl_path=pred_lbl_path,
            label_path=os.path.join(args.data_dir, args.dev_label_file),
            should_remove_de="13" in args.dev_file
            
        )
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    return results

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default=None, type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--image_model_type", default=0, type=int)
    parser.add_argument("--model_name_or_path", default='', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--font_path", default='', type=str)
    parser.add_argument("--data_dir", default="", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")    ### 填充 data 的路径 /..your_path../dorm_project_copy/data
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_file", default="train.pkl", type=str)
    parser.add_argument("--dev_file", default="dev.pkl", type=str)
    parser.add_argument("--dev_label_file", default="dev.lbl.tsv", type=str)
    parser.add_argument("--predict_file", default="test.sighan15.pkl", type=str)
    parser.add_argument("--predict_label_file", default="test.sighan15.lbl.tsv", type=str)
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--order_metric", default='sent-correct-f1', type=str)
    parser.add_argument("--metric_reverse", action='store_true')
    parser.add_argument("--num_save_ckpts", default=5, type=int)
    parser.add_argument("--remove_unused_ckpts", action='store_true')
    
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")     
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    


    parser.add_argument("--add_pinyin_loss", action="store_true")
    parser.add_argument("--pinyin_weight", type=float, default=1.)
    parser.add_argument("--add_pinyin_mask", action="store_true")
    
    parser.add_argument("--use_kl", action="store_true")
    parser.add_argument("--kl_weight", type=float, default=1.)
    parser.add_argument("--second_loss_weight", type=float, default=1.)        
    parser.add_argument("--comment", type=str, default="")


    args = parser.parse_args()
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device


    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    set_seed(args)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    
    config.add_pinyin_loss = args.add_pinyin_loss
    config.pinyin_weight = args.pinyin_weight
    config.add_pinyin_mask = args.add_pinyin_mask
    
    config.use_kl = args.use_kl
    config.kl_weight = args.kl_weight
    config.second_loss_weight = args.second_loss_weight
    

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path, config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    model.tie_cls_weight()


        
    batch_processor = model_class.build_batch

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer, batch_processor)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        
        best_ckpt_dirs = []
        results = {}

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('saved_ckpt-') != -1 else ""
            
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, batch_processor, prefix=prefix)
            best_ckpt_dirs.append((result[args.order_metric], checkpoint))
            
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result) 

        best_ckpt_dirs = sorted(best_ckpt_dirs, reverse=args.metric_reverse)[:args.num_save_ckpts]
        best_ckpt_dirs = [d for v, d in best_ckpt_dirs]

        json.dump(results, open(os.path.join(args.output_dir, 'dev_results.json'), 'w', encoding='utf-8'), indent=4)

        if args.remove_unused_ckpts:
            for checkpoint in checkpoints:
                prefix = checkpoint.split('/')[-1] if checkpoint.find('saved_ckpt-') != -1 else ""
                if len(prefix) != 0 and (checkpoint not in best_ckpt_dirs):
                    logger.info("Deleting ckpt: %s", checkpoint)
                    os.system("rm -rf %s"%checkpoint)

    if args.do_predict and args.local_rank in [-1, 0]:
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        
        args.dev_file = args.predict_file
        args.dev_label_file = args.predict_label_file
        results = {}

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('saved_ckpt-') != -1 else ""
            
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, batch_processor, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

        json.dump(results, open(os.path.join(args.output_dir, 'predict_results.json'), 'w', encoding='utf-8'), indent=4)


