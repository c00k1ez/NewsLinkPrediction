import torch
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
        dim_size = output_tensor.shape[dim]
        output_tensor = output_tensor.mean(dim=dim)
        output_tensor = output_tensor * dim_size
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
        news = self.encoder(news, attention_mask=news_mask)[0] # [batch_size, news_seq_len, encoder_hidden]
        broadcast = self.prehead_dropout(self.pool(broadcast, broadcast_mask)) # [batch_size, encoder_hidden]
        news = self.prehead_dropout(self.pool(news, news_mask)) # [batch_size, encoder_hidden]

        broadcast = self.broadcast_head(broadcast)
        news = self.news_head(news)
        
        return {
            'anchor': broadcast,
            'positive': news
        }


        
