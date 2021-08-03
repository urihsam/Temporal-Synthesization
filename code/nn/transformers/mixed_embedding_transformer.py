import torch
from torch import Tensor
import torch.nn.functional as f
from utils.train_utils import descale_time, extract_incr_time_from_tempo_step
from nn.transformers.utils import scaled_dot_product_attention, position_embedding, feed_forward


class AttentionHead(torch.nn.Module):
    def __init__(self, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.q = torch.nn.Linear(dim_in, dim_k)
        self.k = torch.nn.Linear(dim_in, dim_k)
        self.v = torch.nn.Linear(dim_in, dim_v)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [AttentionHead(dim_in, dim_k, dim_v) for _ in range(num_heads)]
        )
        self.linear = torch.nn.Linear(num_heads * dim_v, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )

class Residual(torch.nn.Module):
    def __init__(self, sublayer: torch.nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = torch.nn.LayerNorm(dimension)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "value" tensor is given last, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[-1] + self.dropout(self.sublayer(*tensors)))



class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self, 
        dim_model: int = 512, 
        num_heads: int = 6, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        attentioned_mask: bool = False
    ):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

        # 
        self.attentioned_mask = attentioned_mask
        if self.attentioned_mask:
            self.attention_m = Residual(
                MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
                dimension=dim_model,
                dropout=dropout,
            )
            self.feed_forward_m = Residual(
                feed_forward(dim_model, dim_feedforward),
                dimension=dim_model,
                dropout=dropout,
            )

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        src = self.attention(src, src, src)
        src = self.feed_forward(src)
        if self.attentioned_mask:
            assert src_mask is not None
            src_mask = self.attention_m(src_mask, src_mask, src_mask)
            src_mask = self.feed_forward_m(src_mask)
        return src, src_mask


class TransformerEncoder(torch.nn.Module):
    def __init__(
        self, 
        time_embedding: Tensor=None,
        attentioned_mask: bool = False,
        num_layers: int = 6,
        dim_feature: int = 9,
        dim_model: int = 512, 
        num_heads: int = 8, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
    ):
        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.time_embedding = time_embedding
        self.attentioned_mask = attentioned_mask
        self.feature2hidden = torch.nn.Linear(dim_feature, dim_model)
        if self.attentioned_mask:
            self.feature2hidden_m = torch.nn.Linear(dim_feature, dim_model)
        self.layers = torch.nn.ModuleList([
            TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout, attentioned_mask)
            for _ in range(num_layers)
        ])

    def forward(self, src: Tensor, time: Tensor, src_mask: Tensor = None) -> Tensor:
        if not self.attentioned_mask and src_mask is not None:
            src = torch.mul(src, src_mask)
        src = self.feature2hidden(src)
        if self.attentioned_mask:
            assert src_mask is not None
            src_mask = self.feature2hidden_m(src_mask)
        seq_len, dimension = src.size(1), src.size(2)
        src += position_embedding(seq_len, dimension)
        src += self.time_embedding(time)
        for layer in self.layers:
            src, src_mask = layer(src, src_mask)
            if self.attentioned_mask:
                src = torch.mul(src, src_mask)

        return src


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(
        self, 
        dim_model: int = 512, 
        num_heads: int = 6, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        attentioned_mask: bool = False
    ):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention_1 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout=dropout,
        )
        self.attention_2 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

        self.attentioned_mask = attentioned_mask
        if self.attentioned_mask:
            self.attention_m_1 = Residual(
                MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
                dimension=dim_model,
                dropout=dropout,
            )
            self.attention_m_2 = Residual(
                MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
                dimension=dim_model,
                dropout=dropout,
            )
            self.feed_forward_m = Residual(
                feed_forward(dim_model, dim_feedforward),
                dimension=dim_model,
                dropout=dropout,
            )

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor = None) -> Tensor:
        tgt = self.attention_1(tgt, tgt, tgt)
        tgt = self.attention_2(memory, memory, tgt)
        tgt = self.feed_forward(tgt)
        if self.attentioned_mask:
            assert tgt_mask is not None
            tgt_mask = self.attention_m_1(tgt_mask, tgt_mask, tgt_mask)
            tgt_mask = self.attention_m_2(memory, memory, tgt_mask)
            tgt_mask = self.feed_forward_m(tgt_mask)
        return tgt, tgt_mask


class TransformerDecoder(torch.nn.Module):
    def __init__(
        self, 
        time_embedding: Tensor=None,
        attentioned_mask: bool = False,
        num_layers: int = 6,
        max_length: int = 50,
        dim_feature: int = 9,
        dim_model: int = 512, 
        dim_time: int = 100,
        num_heads: int = 8, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        use_prob_mask: bool = False,
    ):
        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.time_embedding = time_embedding
        self.attentioned_mask = attentioned_mask
        self.use_prob_mask = use_prob_mask
        self.max_length = max_length
        self.feature_size = dim_feature
        self.dim_model = dim_model
        self.dim_time = dim_time
        self.feature2hidden = torch.nn.Linear(dim_feature, dim_model)
        if self.attentioned_mask:
            self.feature2hidden_m = torch.nn.Linear(dim_feature, dim_model)
        self.layers = torch.nn.ModuleList([
            TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout, attentioned_mask)
            for _ in range(num_layers)
        ])
        self.linear = torch.nn.Linear(dim_model, dim_feature)
        self.linear_m = torch.nn.Linear(dim_model, dim_feature)

    def forward(self, tgt: Tensor, memory: Tensor, time: Tensor, tgt_mask: Tensor = None) -> Tensor:
        if not self.attentioned_mask and tgt_mask is not None:
            tgt = torch.mul(tgt, tgt_mask)
        tgt = self.feature2hidden(tgt)
        if self.attentioned_mask:
            assert tgt_mask is not None
            tgt_mask = self.feature2hidden_m(tgt_mask)
        

        seq_len, dimension = tgt.size(1), tgt.size(2)
        tgt += position_embedding(seq_len, dimension)
        tgt += self.time_embedding(time)
        for layer in self.layers:
            tgt, tgt_mask = layer(tgt, memory, tgt_mask)
            if self.attentioned_mask:
                tgt = torch.mul(tgt, tgt_mask)

        output = torch.nn.functional.sigmoid(self.linear(tgt))
        
        if tgt_mask == None or self.use_prob_mask: # if use_prob_mask, no need to generate mask
            return output, None
        
        if self.attentioned_mask:
            mask =  torch.nn.functional.sigmoid(self.linear_m(tgt_mask))
        else:
            mask =  torch.nn.functional.sigmoid(self.linear_m(tgt))
        return output, mask


    def inference(self, start_feature: Tensor, z: Tensor, start_mask: Tensor = None, prob_mask: Tensor = None, **kwargs):
        if self.use_prob_mask:
            assert prob_mask is not None and start_mask is not None
        
        time_shift = kwargs["time_shift"]
        time_scale = kwargs["time_scale"]
        start_time = kwargs["start_time"]

        z = z.cuda()
        batch_size = z.size(0)
        zs = torch.unbind(z, dim=1) 
        

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
        sequence_length = torch.zeros(batch_size, out=self.tensor()).long()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_length, self.feature_size).fill_(0.0).float()
        gen_masks = self.tensor(batch_size, self.max_length, self.feature_size).fill_(0.0).float()
        
        t=0
        time = start_time.cuda()
        descaled_time = descale_time(time, time_shift, time_scale)
        pos_emb = position_embedding(self.max_length, self.dim_model)[0]
        while(t+1<self.max_length and len(running_seqs)>0):
            batch_size = z.size(0)
            zs = torch.unbind(z, dim=1)
            z_ = zs[t].unsqueeze(dim=1)
            
            #
            if t == 0:
                # input for time step 0
                input_sequence = start_feature.float().cuda() # [batch, feature_size]
                if start_mask == None:
                    input_mask = None
                else:
                    input_mask = start_mask.float().cuda() # [batch, feature_size]
                        
            input_ = input_sequence.unsqueeze(dim=1)
            if not self.attentioned_mask and input_mask is not None:
                input_mask = input_mask.unsqueeze(dim=1)
                input_ = torch.mul(input_, input_mask)
                
            #import pdb; pdb.set_trace()
            input_ = self.feature2hidden(input_)
            if self.attentioned_mask:
                assert input_mask is not None
                input_mask = input_mask.unsqueeze(dim=1)
                input_mask = self.feature2hidden_m(input_mask)

            #input_batch_size, seq_len, dimension = input_.size(0), input_.size(1), input_.size(2)
            input_batch_size = input_.size(0)
            if input_batch_size != batch_size:
                import pdb; pdb.set_trace()
            input_ += pos_emb[t] # position_embedding(seq_len, dimension)
            # time embedding, de-scale before embedding
            input_ += self.time_embedding(descaled_time)
            
            for layer in self.layers:
                try:
                    input_, input_mask = layer(input_, z_, input_mask)
                    if self.attentioned_mask:
                        input_ = torch.mul(input_, input_mask)
                except:
                    import pdb; pdb.set_trace()


            input_sequence = torch.nn.functional.sigmoid(self.linear(input_))
            input_sequence = input_sequence.squeeze(dim=1)
            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t+1)
            
            
            #import pdb; pdb.set_trace()
            #
            if input_mask is not None:
                if self.use_prob_mask:
                    input_mask = sample_mask_from_prob(prob_mask, input_mask.shape[0], input_mask.shape[1])
                    input_mask = input_mask.squeeze(dim=1)
                else:
                    # generate mask
                    # decoder forward pass
                    if self.attentioned_mask:
                        input_mask =  torch.nn.functional.sigmoid(self.linear_m(input_mask))
                    else:
                        input_mask =  torch.nn.functional.sigmoid(self.linear_m(input_))
                    input_mask = input_mask.squeeze(dim=1)
                    # save next input
                    gen_masks = self._save_sample(gen_masks, input_mask, sequence_running, t+1)
            
            

            # get incr time, update time
            time += extract_incr_time_from_tempo_step(input_sequence)
            descaled_time = descale_time(time, time_shift, time_scale)
            
            #
            # update gloabl running sequence
            sequence_length[sequence_running] += 1
            #import pdb; pdb.set_trace()
            # select criteria: sum of feature values > 0 and descaled time < dim_time
            sequence_mask[sequence_running] = torch.logical_and(input_sequence.sum(dim=1) > 0, descaled_time.squeeze(dim=1) < self.dim_time) # ??
            sequence_running = sequence_idx.masked_select(sequence_mask)
            
            # update local running sequences
            running_mask = torch.logical_and(input_sequence.sum(dim=1) > 0, descaled_time.squeeze(dim=1) < self.dim_time) # ??
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                z = z[running_seqs]
                if input_mask is not None:
                    input_mask = input_mask[running_seqs]
                time = time[running_seqs]
                descaled_time = descaled_time[running_seqs]
                
                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()
            #
            t += 1

        output = generations
        
        if start_mask == None or self.use_prob_mask:
            return output, None
        
        mask = gen_masks

        return output, mask
        

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t,:] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
    

class Transformer(torch.nn.Module):
    def __init__(
        self, 
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        max_length: int = 50,
        dim_feature: int = 9,
        dim_model: int = 512, 
        dim_time: int = 100,
        num_heads: int = 6, 
        dim_feedforward: int = 2048, 
        encoder_dropout: float = 0.1, 
        decoder_dropout: float = 0.1, 
        activation: torch.nn.Module = torch.nn.ReLU(),
        attentioned_mask: bool = False,
        use_prob_mask: bool = False,
    ):
        super().__init__()
        self.time_embedding = torch.nn.Embedding(dim_time, dim_model) # an embedding lookup dict with key range from 0 to dim_age-1
        self.encoder = TransformerEncoder(
            time_embedding=self.time_embedding,
            attentioned_mask=attentioned_mask,
            num_layers=num_encoder_layers,
            dim_feature=dim_feature,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=encoder_dropout,
        )
        self.decoder = TransformerDecoder(
            time_embedding=self.time_embedding,
            attentioned_mask=attentioned_mask,
            num_layers=num_decoder_layers,
            max_length=max_length,
            dim_feature=dim_feature,
            dim_model=dim_model,
            dim_time=dim_time,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=decoder_dropout,
            use_prob_mask=use_prob_mask
        )

    def forward(self, src: Tensor, tgt: Tensor, src_time: Tensor, tgt_time: Tensor, 
                src_mask: Tensor = None, tgt_mask: Tensor = None) -> Tensor:
        src = src.float(); src_time=src_time.int(); src_mask = src_mask.float()
        tgt = tgt.float(); tgt_time=tgt_time.int(); tgt_mask = tgt_mask.float()
        #import pdb; pdb.set_trace()
        memory = self.encoder(src, src_time, src_mask)
        output, mask = self.decoder(tgt, memory, tgt_time, tgt_mask)
        # build a target prob tensor
        if tgt_mask == None:
            p_input = tgt
        else:
            p_input = torch.mul(tgt, tgt_mask)
        return memory, p_input, output, mask


    def compute_mask_loss(self, output_mask, mask, type="xent"):
        if type == "mse":
            loss = torch.mean(torch.sum(torch.square(output_mask-mask), dim=(1, 2)))
        elif type == "xent":
            entropy = torch.mul(mask, torch.log(output_mask + 1e-12)) + torch.mul((1-mask), torch.log((1-output_mask + 1e-12)))
            loss = torch.mean(torch.sum(-1.0*entropy, dim=(1, 2)))
        else:
            raise "Wrong loss type"
        return loss
    

    def compute_recon_loss(self, output, target, output_mask=None, mask=None, type="xent"):
        if type == "mse":
            loss_1 = torch.mean(torch.sum(torch.square(output-target), dim=(1, 2)))
        elif type == "xent":
            entropy = torch.mul(target, torch.log(output + 1e-12)) + torch.mul((1-target), torch.log((1-output + 1e-12)))
            loss_1 = torch.mean(torch.sum(-1.0*entropy, dim=(1, 2)))
        else:
            raise "Wrong loss type"
        if output_mask is not None:
            output = torch.mul(output, output_mask)
        if mask is not None:
            target = torch.mul(target, mask)
        if type == "mse":
            loss_2 = torch.mean(torch.sum(torch.square(output-target), dim=(1, 2)))
        elif type == "xent":
            entropy = torch.mul(target, torch.log(output + 1e-12)) + torch.mul((1-target), torch.log((1-output + 1e-12)))
            loss_2 = torch.mean(torch.sum(-1.0*entropy, dim=(1, 2)))  
        else:
            raise "Wrong loss type"
 
        return (loss_1 + loss_2)/2