import torch
import torch.nn as nn

from ...attention import CausalAttention

class ScratchGPTEmbedding(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings = config.vocab_size, 
            embedding_dim = config.embedding_dim, 
            device = config.device)
        
        self.position_embedding = nn.Embedding(
            num_embeddings = config.context_length, 
            embedding_dim = config.embedding_dim, 
            device = config.device)
        
        self.dropout = nn.Dropout(p = config.dropout)

    def forward(self, input_ids):

        batch_size, num_tokens = input_ids.shape[0], input_ids.shape[1]

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(torch.arange(num_tokens).to(input_ids.device))
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings

class ScratchGPTFFN(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(
                in_features = config.embedding_dim, 
                out_features = 4 * config.embedding_dim, 
                device = config.device),
            nn.GELU(),
            nn.Linear(
                in_features = 4 * config.embedding_dim, 
                out_features = config.embedding_dim, 
                device = config.device))

    def forward(self, hidden_states):
        return self.layers(hidden_states)

class ScratchGPTLayer(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.ln1 = nn.LayerNorm(normalized_shape = config.embedding_dim, device = config.device)
        self.attention = CausalAttention(config)

        self.ln2 = nn.LayerNorm(normalized_shape = config.embedding_dim, device = config.device)
        self.ffn = ScratchGPTFFN(config)

        self.dropout = nn.Dropout(p = config.dropout)

    def forward(self, hidden_states):

        # pass through attention layers first
        residuals = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + residuals

        # then through the feed-forward network
        residuals = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + residuals

        return hidden_states

class ScratchGPT(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.embedding_layer = ScratchGPTEmbedding(config)
        self.gpt_layers = nn.Sequential(*[ScratchGPTLayer(config) for n in range(config.num_layers)])
        self.final_ln = nn.LayerNorm(
            normalized_shape = config.embedding_dim, 
            device = config.device)

    def forward(self, input_ids):

        hidden_states = self.embedding_layer(input_ids)
        hidden_states = self.gpt_layers(hidden_states)
        hidden_states = self.final_ln(hidden_states)

        return hidden_states

class ScratchGPTLMModel(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.gpt = ScratchGPT(config)
        self.lm_head = nn.Linear(
          config.embedding_dim, 
          config.vocab_size, 
          bias = False, 
          device = config.device)

    def forward(self, input_ids):

        hidden_states = self.gpt(input_ids)
        logits = self.lm_head(hidden_states)

        return logits
