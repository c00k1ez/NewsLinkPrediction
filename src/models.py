import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

class MaskedAveragePooling(torch.nn.Module):
    def __init__(self):
        super(MaskedAveragePooling, self).__init__()
    
    def forward(self, output_tensor, mask=None, dim=1):
        if mask is None:
            mask = torch.ones(output_tensor.shape[:-1]).type_as(output_tensor)
            mask_repeated = torch.ones(output_tensor.shape).type_as(output_tensor)
        else:
            mask_repeated = mask.unsqueeze(-1).repeat([1, 1, output_tensor.shape[2]])
            mask_repeated = mask_repeated.type_as(output_tensor)
        output_tensor = output_tensor * mask_repeated
        #dim_size = output_tensor.shape[dim]
        output_tensor = output_tensor.sum(dim=dim)
        output_tensor[output_tensor.isnan()] = 0.
        current_length = mask.sum(dim=-1).unsqueeze(-1).repeat([1, output_tensor.shape[-1]])
        current_length = current_length.type_as(mask)
        output_tensor = output_tensor / current_length
        return output_tensor


class BaselineSiameseNetwork(torch.nn.Module):
    def __init__(
            self,
            encoder: transformers.PreTrainedModel,
            encoder_hidden: int,
            output_dim: int,
            dropout: int = 0.2
        ):
        super(BaselineSiameseNetwork, self).__init__()
        self.encoder = encoder
        self.encoder_hidden = encoder_hidden
        self.output_dim = output_dim
        self.pool = MaskedAveragePooling()
        self.prehead_dropout = torch.nn.Dropout(dropout)
        self.broadcast_head = torch.nn.Linear(encoder_hidden, output_dim)
        self.news_head = torch.nn.Linear(encoder_hidden, output_dim)

    def forward(self, batch):
        broadcast = batch['broadcast']
        broadcast_mask = batch['broadcast_mask']
        news = batch['news']
        news_mask = batch['news_mask']

        broadcast = self.encoder(broadcast, attention_mask=broadcast_mask)[0] # [batch_size, br_seq_len, encoder_hidden]
        broadcast = self.prehead_dropout(self.pool(broadcast, broadcast_mask)) # [batch_size, encoder_hidden]
        
        news = self.encoder(news, attention_mask=news_mask)[0] # [batch_size, news_seq_len, encoder_hidden]
        news = self.prehead_dropout(self.pool(news, news_mask)) # [batch_size, encoder_hidden]

        broadcast = self.broadcast_head(broadcast)
        news = self.news_head(news)
        
        return {
            'anchor': broadcast,
            'positive': news
        }


class SiameseNetwork(torch.nn.Module):
    def __init__(
            self,
            encoder: transformers.PreTrainedModel,
            encoder_hidden: int,
            output_dim: int,
            n_chunks: int = 4,
            chunk_size: int = 512,
            dropout: int = 0.2
        ):
        super(SiameseNetwork, self).__init__()
        self.encoder = encoder
        self.encoder_hidden = encoder_hidden
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.output_dim = output_dim
        self.pool = MaskedAveragePooling()
        self.prehead_dropout = torch.nn.Dropout(dropout)
        self.broadcast_head = torch.nn.Linear(encoder_hidden * n_chunks, output_dim)
        self.news_head = torch.nn.Linear(encoder_hidden, output_dim)

    def forward(self, batch):
        broadcast = batch['broadcast']
        broadcast_mask = batch['broadcast_mask']
        news = batch['news']
        news_mask = batch['news_mask']

        assert self.chunk_size * self.n_chunks == broadcast.shape[1]
        broadcast = torch.split(broadcast, self.chunk_size, dim=1)
        broadcast_mask = torch.split(broadcast_mask, self.chunk_size, dim=1)

        encoded_broadcast = []
        for br, attn_mask in zip(broadcast, broadcast_mask):
            encoded_broadcast.append(self.encoder(br, attention_mask=attn_mask)[0])
        # encoded_broadcast - list of 4 tensors with shape [batch_size, chunk_size, 768]
        assert len(encoded_broadcast) == 4
        pooled = []
        for tensor in encoded_broadcast:
            pooled.append(self.pool(tensor))

        assert len(pooled) == 4
        # pooled - list of 4 tensors with shape [batch_size, 768]
        pooled = torch.nn.functional.relu(torch.cat(pooled, dim=1)) # shape [batch_size, 3072]
        pooled = self.prehead_dropout(pooled)

        news = self.encoder(news, attention_mask=news_mask)[0] # [batch_size, news_seq_len, encoder_hidden]
        news = self.prehead_dropout(self.pool(news, news_mask)) # [batch_size, encoder_hidden]

        pooled = self.broadcast_head(pooled)
        news = self.news_head(news)
        
        return {
            'anchor': pooled,
            'positive': news
        }


class MemoryAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, proj_output=None, in_dim=None, dropout=0.1):
        super(MemoryAttention, self).__init__()
        assert d_model % n_heads == 0

        self.hid = d_model
        if in_dim is None:
            in_dim = d_model
        self.K = nn.Linear(in_dim, d_model)
        self.Q = nn.Linear(in_dim, d_model)
        self.V = nn.Linear(in_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        if proj_output is None:
            proj_output = d_model
        self.projection = nn.Linear(d_model, proj_output)
        self.scale = (d_model / n_heads) ** 0.5
        self.n_heads = n_heads

    def forward(self, query, key, value, attention_mask=None, return_attention_weights=False, mask_for_kv=False):
        # attention_mask shape: [bs, seq_len], 1 - non masked, 0 - masked
        batch_size, in_seq_len, in_hid = query.shape
        mid_seq_len = key.shape[1]

        h_dim = self.hid // self.n_heads

        k, q, v = self.K(key), self.Q(query), self.V(value)
        k = self.dropout(k)
        q = self.dropout(q)
        v = self.dropout(v)

        # (bs, seq, hid) -> (seq, bs * n_heads, h_dim) -> (bs * n_heads, seq, h_dim)
        k = k.transpose(0, 1).contiguous().view(mid_seq_len, batch_size * self.n_heads, h_dim).transpose(0, 1)
        q = q.transpose(0, 1).contiguous().view(in_seq_len, batch_size * self.n_heads, h_dim).transpose(0, 1)
        v = v.transpose(0, 1).contiguous().view(mid_seq_len, batch_size * self.n_heads, h_dim).transpose(0, 1)

        if attention_mask is not None:
            mask_len = attention_mask.shape[1]
            attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, self.hid)
            attention_mask = attention_mask.transpose(0, 1).contiguous().view(mask_len, batch_size * self.n_heads, h_dim).transpose(0, 1)
            attention_mask = attention_mask[:, :, 0]
        
        output = ()
        non_weight_attn_scores = q @ k.transpose(1, 2) / self.scale

        if attention_mask is not None:
            if mask_for_kv:
                seq_len = in_seq_len
            else:
                seq_len = mid_seq_len
            attn_mask = attention_mask.unsqueeze(-1).repeat(1, 1, seq_len)
            if mask_for_kv:
                attn_mask = attn_mask.transpose(1, 2)
            
            #attn_mask = attn_mask.transpose(1, 2) * attn_mask
            attn_mask = attn_mask.float().masked_fill(attn_mask == 0., -100000000000)
            attn_mask = attn_mask.float().masked_fill(attn_mask == 1, 0)
            non_weight_attn_scores = non_weight_attn_scores + attn_mask
        
        attn_scores = F.softmax(non_weight_attn_scores, dim=-1)
        if attention_mask is not None:
            attn_mask = attn_mask.float().masked_fill(attn_mask == 0., 1)
            attn_scores = attn_scores * attn_mask.float().masked_fill(attn_mask == -100000000000., 0.)
        
        attn = attn_scores @ v
        attn = attn.transpose(0, 1).contiguous().view(in_seq_len, batch_size, self.hid).transpose(0, 1)
        attn = self.projection(attn)
        output = (attn, )
        if return_attention_weights:
            output = output + (attn_scores.unsqueeze(1).view(batch_size, self.n_heads, in_seq_len, -1), )
        return output


class MemoryEncoder(nn.Module):
    def __init__(
            self,
            attn_hidden,
            n_heads=4,
            update_coeffs=(0.75, 0.25),
            input_size=None,
            output_size=None,
            mem_elements=16,
            k=4
        ):
        super(MemoryEncoder, self).__init__()
        self.update_coeffs = update_coeffs
        if input_size is None:
            input_size = attn_hidden
        if output_size is None:
            output_size = attn_hidden
        self.in_projection = nn.Linear(input_size, input_size // k)        
        self.memory_attention = MemoryAttention(d_model=attn_hidden, n_heads=n_heads, proj_output=input_size//k, in_dim=input_size//k)
        self.seq_attention = MemoryAttention(d_model=attn_hidden, n_heads=n_heads, proj_output=output_size, in_dim=input_size//k)

        self.memory = nn.Embedding(mem_elements, input_size // k)
        self.mem_elements = mem_elements
        self.ln = nn.LayerNorm(output_size)
    
    def forward(self, sequence, attention_mask=None, memory=None, use_mem=16, first_step=False, return_attention_weights=False):
        assert use_mem >= 1 and use_mem <= self.mem_elements
        batch_size = sequence.shape[0]
        if first_step:
            mem_inds = torch.arange(0, use_mem).type_as(sequence).long()
            mem = self.memory(mem_inds).unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            mem = memory
        
        in_sequence = self.in_projection(sequence)
        # step 1: attend to sequnce
        mem_outputs = self.memory_attention(
            query=mem,
            key=in_sequence,
            value=in_sequence,
            attention_mask=attention_mask,
            return_attention_weights=return_attention_weights,
            mask_for_kv=True
        )
        # step 2: update memory
        mem = self.update_coeffs[0] * mem + self.update_coeffs[1] * mem_outputs[0]

        # step 3: attend to memory
        seq_outputs = self.seq_attention(
            query=in_sequence,
            key=mem,
            value=mem,
            attention_mask=attention_mask,
            return_attention_weights=return_attention_weights,
            mask_for_kv=False
        )

        # step 4: update sequence
        sequence = self.ln(sequence + seq_outputs[0])
        outputs = (sequence, mem)
        if return_attention_weights:
            outputs = outputs + ({
                'mem_attention_map': mem_outputs[1],
                'seq_attention_map': seq_outputs[1]
            }, )
        return outputs
    

class SiameseMemoryNetwork(nn.Module):
    def __init__(
            self,
            encoder: transformers.PreTrainedModel,
            encoder_hidden: int,
            output_dim: int,
            n_chunks: int = 4,
            chunk_size: int = 512,
            dropout: int = 0.2,
            attention_hidden: int = 128,
            n_heads=4,
            update_coeffs=(0.75, 0.25),
            input_size=758,
            output_size=768,
            mem_elements=16,
            k=4
        ):
        super(SiameseMemoryNetwork, self).__init__()
        self.encoder = encoder
        self.memory_encoder = MemoryEncoder(
            attn_hidden=attention_hidden,
            n_heads=n_heads,
            update_coeffs=tuple(update_coeffs),
            input_size=input_size,
            output_size=output_size,
            mem_elements=mem_elements,
            k=k
        )
        self.encoder_hidden = encoder_hidden
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.output_dim = output_dim
        self.pool = MaskedAveragePooling()
        self.prehead_dropout = nn.Dropout(dropout)
        self.broadcast_head = nn.Linear(encoder_hidden * n_chunks, output_dim)
        self.news_head = nn.Linear(encoder_hidden, output_dim)

    def forward(self, batch,return_mem_attention_weights=False):
        broadcast = batch['broadcast']
        broadcast_mask = batch['broadcast_mask']
        news = batch['news']
        news_mask = batch['news_mask']

        assert self.chunk_size * self.n_chunks == broadcast.shape[1]
        broadcast = torch.split(broadcast, self.chunk_size, dim=1)
        broadcast_mask = torch.split(broadcast_mask, self.chunk_size, dim=1)

        encoded_broadcast = []
        mem_attn_weights = []
        first_step = True
        prev_memory = None
        for br, attn_mask in zip(broadcast, broadcast_mask):
            enc_output = self.encoder(br, attention_mask=attn_mask)[0]
            mem_outputs = self.memory_encoder(
                enc_output,
                memory=prev_memory,
                attention_mask=attn_mask,
                first_step=first_step,
                return_attention_weights=return_mem_attention_weights
            )
            first_step = False
            enc_output, prev_memory = mem_outputs[:2]
            encoded_broadcast.append(enc_output)
            if return_mem_attention_weights:
                mem_attn_weights.append(mem_outputs[2])

        # encoded_broadcast - list of 4 tensors with shape [batch_size, chunk_size, 768]
        assert len(encoded_broadcast) == 4
        pooled = []

        for tensor in encoded_broadcast:
            pooled.append(self.pool(tensor))

        assert len(pooled) == 4
        # pooled - list of 4 tensors with shape [batch_size, 768]
        pooled = torch.nn.functional.relu(torch.cat(pooled, dim=1)) # shape [batch_size, 3072]
        pooled = self.prehead_dropout(pooled)

        news = self.encoder(news, attention_mask=news_mask)[0] # [batch_size, news_seq_len, encoder_hidden]
        news = self.prehead_dropout(self.pool(news, news_mask)) # [batch_size, encoder_hidden]

        pooled = self.broadcast_head(pooled)
        news = self.news_head(news)
        
        if return_mem_attention_weights:
            return ({
                'anchor': pooled,
                'positive': news
            }, mem_attn_weights)
        
        return {
            'anchor': pooled,
            'positive': news
        }