from transformers.modeling_bert import *
import torch.nn.functional as F


class PinyinBertModel(BertPreTrainedModel):

    def __init__(self, config):
        super(PinyinBertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()
        

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, other_embeds=None, embedding_output=None, text_length=None):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(torch.long)  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
                """在这里添加 pinyin 看不见文本,文本可看见拼音"""
                if text_length is not None:
                    seq_len = extended_attention_mask.size()[-1]
                    extended_attention_mask = extended_attention_mask.repeat(1,1, seq_len, 1)
                    extended_attention_mask[:, :, text_length:, :text_length] = 0
                
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape, attention_mask.shape))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError("Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(encoder_hidden_shape,
                                                                                                                               encoder_attention_mask.shape))

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        if embedding_output is None:
            embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, other_embeds=other_embeds)
        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class DORM(BertPreTrainedModel):
    def __init__(self, config):
        super(DORM, self).__init__(config)

        self.vocab_size = config.vocab_size
        self.bert = PinyinBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        
        self.add_pinyin_loss = config.add_pinyin_loss if hasattr(config, "add_pinyin_loss") else False
        self.pinyin_weight = config.pinyin_weight if hasattr(config, "pinyin_weight") else 1
        self.add_pinyin_mask = config.add_pinyin_mask if hasattr(config, "add_pinyin_mask") else True
        
        ## 是否加入自蒸馏模块
        self.use_kl = config.use_kl if hasattr(config, "use_kl") else False
        self.kl_weight = config.kl_weight if hasattr(config, "kl_weight") else 1
        
        ## 自蒸馏前向传播的 second-loss 权重为多少
        self.second_loss_weight = config.second_loss_weight if hasattr(config, "second_loss_weight") else 1
        
        self.init_weights()

    def tie_cls_weight(self):
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    @staticmethod
    def build_batch(batch, tokenizer):
        return batch
    
    
    def get_pinyin_bert_sequence_output(self, input_ids, attention_mask, sm_ids, ym_ids, pinyin_position_ids, pinyin_masks, text_length=None ):
        self.bert_embedding = self.bert.embeddings
        src_input_embeds = self.bert_embedding(input_ids=input_ids) # 有了 positon\type\ln\dropout
        sm_embeds = self.bert_embedding.word_embeddings(sm_ids)
        ym_embeds = self.bert_embedding.word_embeddings(ym_ids)
        pinyin_input_embeds = self.bert_embedding(position_ids=pinyin_position_ids, inputs_embeds=sm_embeds+ym_embeds, token_type_ids=pinyin_masks) # 有了 positon\type\ln\dropout

        embedding_output = src_input_embeds * (1-pinyin_masks).unsqueeze(-1).repeat(1, 1, self.config.hidden_size)
        embedding_output += pinyin_input_embeds * pinyin_masks.unsqueeze(-1).repeat(1, 1, self.config.hidden_size)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, embedding_output=embedding_output, text_length=text_length)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return outputs, sequence_output, logits    
    
    def forward(self, batch):
        
        input_ids = batch['src_idx']
        attention_mask = batch['masks']
        loss_mask = batch['loss_masks']
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None
        
        pinyin_loss_mask = batch["pinyin_loss_masks"] if "pinyin_loss_masks" in batch else None
        pinyin_label_ids = batch["pinyin_tgt_idx"] if "pinyin_tgt_idx" in batch else None

        pinyin_masks = batch["pinyin_masks"]
        pinyin_position_ids = batch["pinyin_position_ids"]
        sm_ids = batch["sm_ids"]
        ym_ids = batch["ym_ids"]
        
        text_length = None
        if self.add_pinyin_mask:
            text_length = torch.max( torch.sum( batch["loss_masks"], dim=-1) ).item() + 2 # text_length 为 cls 到 最长 sep 的长度。
        
        outputs, sequence_output, logits = self.get_pinyin_bert_sequence_output(input_ids=input_ids,
                                                                                attention_mask=attention_mask,
                                                                                sm_ids=sm_ids,
                                                                                ym_ids=ym_ids,
                                                                                pinyin_position_ids=pinyin_position_ids,
                                                                                pinyin_masks=pinyin_masks,
                                                                                text_length=text_length)
        
        
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        pinyin_loss = None
        kl_loss = None
        
        if label_ids is not None:
            loss_fct = CrossEntropyLoss().to(input_ids.device)
            # Only keep active parts of the loss
            active_loss = loss_mask.view(-1) == 1
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            if self.add_pinyin_loss:
                pinyin_active_loss = pinyin_loss_mask.view(-1) == 1
                pinyin_active_logits = logits.view(-1, self.vocab_size)[pinyin_active_loss]
                pinyin_active_labels = pinyin_label_ids.view(-1)[pinyin_active_loss]
                pinyin_loss = loss_fct(pinyin_active_logits, pinyin_active_labels)
                pinyin_loss = self.pinyin_weight * pinyin_loss
                loss += pinyin_loss
            
            if self.training and self.use_kl:

                max_len = torch.max( torch.sum( batch["loss_masks"], dim=-1)) + 2
                second_input_ids = batch["src_idx"][:, :max_len]
                    
                second_attention_mask = batch["masks"][:, :max_len]
                second_loss_mask = batch["loss_masks"][:, :max_len]
                second_active_loss = second_loss_mask.reshape(-1) == 1
                second_attention_mask[ second_input_ids == 0] = 0
                
                second_outputs = self.bert(second_input_ids, second_attention_mask)    
                second_sequence_output = self.dropout( second_outputs[0] )
                second_logits = self.classifier( second_sequence_output )
                second_active_logits = second_logits.view(-1, self.vocab_size)[second_active_loss]
                second_loss = loss_fct(second_active_logits, active_labels)
                
                ## 自蒸馏的 loss                
                kl_loss = self.calculate_kl_loss(second_active_logits, active_logits)   
                kl_loss += self.calculate_kl_loss(active_logits, second_active_logits)
                kl_loss = kl_loss /2 
                
                kl_loss = self.kl_weight * kl_loss + self.second_loss_weight * second_loss
                loss += kl_loss           
                
            
            outputs = (loss,) + outputs + (pinyin_loss, kl_loss)
                
        return outputs 

    
    def calculate_kl_loss(self, teacher_logits, student_logits):

        kl_loss = torch.nn.functional.kl_div(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction='none')
        kl_loss = torch.sum(kl_loss, dim=-1)
        kl_loss = kl_loss.mean()
        return kl_loss
