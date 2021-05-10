import copy
from typing import Optional, List

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor

class Transformer(nn.Layer):
    '''
        Top Level class
    '''
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 normalize_before=False,
                 return_intermediate=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, normalize_before)
        encoder_norm  = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder  = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, normalize_before)
        decoder_norm  = nn.LayerNorm(d_model)
        self.decoder  = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate)

        self._reset_parameters()
        self.d_model  = d_model
        self.nhead    = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.initializer.XavierUniform(p)

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = paddle.flatten(src, 2)
        src = paddle.transpose(src, perm=[0, 2, 1])
        #print("src.shape:")
        #print(src.shape)
        pos_embed = paddle.flatten(pos_embed, 2)
        pos_embed = paddle.transpose(pos_embed, perm=[0, 2, 1])
        #print("pos_embed.shape:")
        #print(pos_embed.shape)
        query_embed = paddle.unsqueeze(query_embed, axis=0)
        query_embed = paddle.broadcast_to(query_embed, shape=[bs, query_embed.shape[1], query_embed.shape[2]])
        #print("query_embed.shape:")
        #print(query_embed.shape)
        mask        = paddle.cast(mask, dtype='float32')
        mask        = paddle.flatten(mask, 1)
        #print("mask.shape:")
        #print(mask.shape)
        tgt         = paddle.zeros_like(query_embed)
        #print("tgt.shape:")
        #print(tgt.shape)
        memory      = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs          = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        #print("hs.shape")
        #print(hs.shape)
        hs          = paddle.transpose(hs, perm=[0, 2, 1, 3])
        memory      = paddle.transpose(memory, perm=[1, 2, 0])
        memory      = paddle.reshape(memory, shape=[bs, c, h, w])
        return hs, memory


class TransformerEncoder(nn.Layer):
    '''
        Top Level Encoder class
    '''
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm   = norm
        

    def forward(self, src,
                mask: Optional[paddle.Tensor] = None,
                src_key_padding_mask: Optional[paddle.Tensor] = None,
                pos: Optional[paddle.Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        
        if self.norm is not None:
            output = self.norm(output)

        return output 


class TransformerDecoder(nn.Layer):
    '''
        Top Level Decoder class
    '''
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm   = norm
        self.return_intermediate = return_intermediate

    def forward(self,tgt, memory, 
                tgt_mask:Optional[paddle.Tensor] = None,
                memory_mask: Optional[paddle.Tensor] = None,
                tgt_key_padding_mask: Optional[paddle.Tensor] = None,
                memory_key_padding_mask: Optional[paddle.Tensor] = None,
                pos: Optional[paddle.Tensor] = None,
                query_pos: Optional[paddle.Tensor] = None):
        '''
            Now, we don't know,whay the decoder class will add a new axis in result...
        '''
        output = tgt
        intermediate = []
        
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return paddle.stack(intermediate)
        
        return paddle.unsqueeze(output, axis=0)
        

class TransformerDecoderLayer(nn.Layer):
    '''
        Basic Decoder Layer of a Encoder
    '''
    def __init__(self,d_model, nhead, dim_feedforward=2048, dropout=0.1, normalize_before=False):
        super().__init__()
        self.attn = nn.MultiHeadAttention(d_model, nhead, dropout)
        self.multihead_attn = nn.MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.gelu
        self.normalize_before = normalize_before

    def with_pos_embd(self, tensor, pos: Optional[paddle.Tensor]):
        return tensor if pos is None else tensor+pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[paddle.Tensor] = None,
                     memory_mask: Optional[paddle.Tensor] = None,
                     tgt_key_padding_mask: Optional[paddle.Tensor] = None,
                     memory_key_padding_mask: Optional[paddle.Tensor] = None,
                     pos: Optional[paddle.Tensor] = None,
                     query_pos: Optional[paddle.Tensor] = None):
        q = k = self.with_pos_embd(tgt, query_pos)
        tgt2 = self.attn(q, k, value=tgt, attn_mask=tgt_mask)[0]
        tgt  = tgt + self.dropout1(tgt2)
        tgt  = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embd(tgt, query_pos),
                                   key=self.with_pos_embd(memory, pos),
                                   value=memory, attn_mask=memory_mask)[0]
        tgt  = tgt + self.dropout2(tgt2)
        tgt  = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt  = tgt + self.dropout3(tgt2)
        tgt  = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                     tgt_mask: Optional[paddle.Tensor] = None,
                     memory_mask: Optional[paddle.Tensor] = None,
                     tgt_key_padding_mask: Optional[paddle.Tensor] = None,
                     memory_key_padding_mask: Optional[paddle.Tensor] = None,
                     pos: Optional[paddle.Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embd(tgt2, query_pos)
        tgt2 = self.attn(q, k, value=tgt2, attn_mask=tgt_mask)[0]
        tgt  = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embd(tgt2, query_pos),
                                   key=self.with_pos_embd(memory, pos),
                                   value=memory, attn_mask=memory_mask)[0]
        tgt  = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt  = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                     tgt_mask: Optional[paddle.Tensor] = None,
                     memory_mask: Optional[paddle.Tensor] = None,
                     tgt_key_padding_mask: Optional[paddle.Tensor] = None,
                     memory_key_padding_mask: Optional[paddle.Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerEncoderLayer(nn.Layer):
    '''
        Basic Encoder Layer of a Decoder.
        d_model : the dimension of Decoder.            (int)
        nhead   : the number of self-attention head.   (int)
        dim_feedforward: the dimension of FFN          (int)
        dropout : the precent of dropout               (float)
        normalize_before: whether process normalize of input    (boolean)
    '''
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, normalize_before=False):
        super().__init__()
        '''
            In paddle's multiheadattention, forward() have five paramaters
            1. query           shape[batch_size, query_length, embed_dim]
            2. key             shape[batch_size, key_length  , kdim]
            3. value           shape[batch_size, value_length, vdim]
            4. attn_mask       shape[batch_size, n_head, sequence_length, sequence_length]
            5. cache        
            But in this file, the top level transformer class only pass the image's mask as src_key_mask_attn rather than mask...
            So, In this class's forward function, the mask is none.
        '''
        self.attn    = nn.MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu
        self.normalize_before = normalize_before
        
    def with_pos_embd(self, tensor, pos:Optional[paddle.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                src,
                src_mask: Optional[paddle.Tensor] = None,
                src_key_padding_mask: Optional[paddle.Tensor] = None,
                pos: Optional[paddle.Tensor] = None):
        '''
            First flow attn, and then process the layernorm
        '''
        q = k = self.with_pos_embd(src, pos)
        src2 = self.attn(q, k, value=src)[0]
        src  = src + self.dropout1(src2)
        src  = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src  = src + self.dropout2(src2)
        src  = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[paddle.Tensor] = None,
                    src_key_padding_mask: Optional[paddle.Tensor] = None,
                    pos: Optional[paddle.Tensor] = None):
        '''
            First process the layernorm, and then process the attn.
        '''
        src2 = self.norm1(src)
        q = k = self.with_pos_embd(src2, pos)
        src2 = self.attn(q, k, value=src2, attn_mask=src_mask)[0]
        src  = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src  = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[paddle.Tensor] = None,
                src_key_padding_mask: Optional[paddle.Tensor] = None,
                pos: Optional[paddle.Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

def _get_clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for i in range(N)])

def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        normalize_before=args.pre_norm,
        return_intermediate=True
    )