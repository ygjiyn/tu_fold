import torch
from torch import nn

import math

from my_transformer import TransformerEncoder


# PositionalEncoding from: 
# https://github.com/pytorch/examples/blob/main/word_language_model/model.py
# [Modify] input shape [batch size, sequence length, embed dim]
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [Modify]
        # pe = pe.unsqueeze(0).transpose(0, 1)
        # pe: batch 1, max_len, d_model
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        # [Modify]
        # input shape [batch size, sequence length, embed dim]
        # x = x + self.pe[:x.size(0), :]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)
        self.conv_layer_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv_layer_1(x)
        out = self.batch_norm_1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv_layer_2(out)
        out = self.batch_norm_2(out)
        out = torch.nn.functional.relu(out)
        return out


class UNetUpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample_layer = nn.Upsample(scale_factor=2)
        self.conv_layer = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.up_sample_layer(x)
        out = self.conv_layer(out)
        out = self.batch_norm(out)
        out = torch.nn.functional.relu(out)
        return out


class UNetEncoder(nn.Module):
    def __init__(self, channels_list):
        '''
        `channels_list` should be a list contains channel numbers of
        the in_channels of UNet encoder and out_channels of each conv block.

        i.e., [in_channels, out_channels_1, out_channels_2, ...]
        '''
        super().__init__()
        self.contracting_path = nn.ModuleList([UNetConvBlock(channels_list[0], channels_list[1])])
        for i in range(1, len(channels_list) - 1):
            self.contracting_path.append(nn.Sequential(
                nn.MaxPool2d(2),
                UNetConvBlock(channels_list[i], channels_list[i + 1])
            ))
        
    def forward(self, x):
        skipped_connections = []
        for conv_block in self.contracting_path:
            x = conv_block(x)
            skipped_connections.append(x)
        # exclude the output of the last conv block from skipped_connections
        return x, skipped_connections[:-1]
    

class UNetDecoder(nn.Module):
    def __init__(self, channels_list):
        '''
        `channels_list` should be a list contains channel numbers of
        the out_channels of each conv block and the out_channels of UNet decoder

        i.e., [out_channels_n, out_channels_n-1, ..., out_channels]
        '''
        super().__init__()
        self.expansive_path = nn.ModuleList([])
        for i in range(len(channels_list) - 1):
            self.expansive_path.append(nn.ModuleList([
                UNetUpConv(channels_list[i], channels_list[i + 1]),
                UNetConvBlock(channels_list[i + 1] * 2, channels_list[i + 1])
            ]))

    def forward(self, x, skipped_connections):
        skipped_connections.reverse()
        for i in range(len(self.expansive_path)):
            up_sample, conv = self.expansive_path[i]
            x = up_sample(x)
            # x: batch_size, channels_num, h, w
            
            if x.shape[-1] < skipped_connections[i].shape[-1]:
                padded_x = torch.zeros((x.shape[0], x.shape[1], x.shape[2] + 1, x.shape[3] + 1)).float().to(x.device)
                padded_x[:, :, :-1, :-1] = x
                x = padded_x
            x = conv(torch.cat((x, skipped_connections[i]), dim=1))
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        decoder_out_channels = 32
        encoder_channels_list = [in_channels, decoder_out_channels, 64, 128, 256, 512]
        decoder_channels_list = [512, 256, 128, 64, decoder_out_channels]

        # decoder_out_channels = 64
        # encoder_channels_list = [in_channels, decoder_out_channels, 128, 256, 512, 1024]
        # decoder_channels_list = [1024, 512, 256, 128, decoder_out_channels]

        self.encoder = UNetEncoder(encoder_channels_list)
        self.decoder = UNetDecoder(decoder_channels_list)

        self.out = nn.Conv2d(decoder_out_channels, out_channels, 1, padding=0)

    def forward(self, x):
        latent, skipped_connections = self.encoder(x)
        latent = self.decoder(latent, skipped_connections)
        y = self.out(latent)
        return y


class Model(nn.Module):
    def __init__(self, 
                 num_embeddings, padding_idx, 
                 d_model, h, d_ff, N, seq_len):
        super().__init__()
        self.pos_encoder = PositionalEncoding(
            d_model=d_model, 
            max_len=seq_len
        )
        self.input_embedding = nn.Embedding(
            num_embeddings=num_embeddings, 
            embedding_dim=d_model,
            padding_idx=padding_idx
        )
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model, 
            n_head=h, 
            n_layer=N, 
            d_ff=d_ff, 
        )
        self.conv_net = UNet(2 * d_model, 1)

        mask_no_sharp_loops = torch.ones((seq_len, seq_len), dtype=torch.float)
        for i in range(seq_len):
            l_idx = i - 3
            if l_idx < 0:
                l_idx = 0
            r_idx = i + 3
            # no problem without the following if block
            # we just want to slice explicitly to make it clear
            if r_idx > seq_len - 1:
                r_idx = seq_len - 1
            # note that l_idx (inclusive) r_idx (exclusive) when slicing
            r_idx += 1
            mask_no_sharp_loops[i, l_idx:r_idx] = 0.0
        self.register_buffer('mask_no_sharp_loops', mask_no_sharp_loops)

    def forward(self, x: torch.Tensor):
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # x: batch_size, seq_len, d_model
        batch_size, seq_len, d_model = x.size()

        x = x.unsqueeze(1)
        # x: batch_size, 1, seq_len, d_model
        x = x.repeat((1, seq_len, 1, 1))
        # x: batch_size, seq_len, seq_len, d_model
        # pairwise concat
        x = torch.cat([x, x.transpose(1, 2)], dim=3)
        # x: batch_size, seq_len, seq_len, d_model * 2
        x = x.permute((0, 3, 1, 2))
        # x: batch_size, d_model * 2, seq_len, seq_len, for conv

        x = self.conv_net(x)
        # x: batch_size, 1, seq_len, seq_len

        x = x.squeeze(1)
        # x: batch_size, seq_len, seq_len

        # make the output symmetric
        x = (x + x.transpose(1, 2)) / 2
        
        if self.training:
            return x
        else:
            row_largest_mat = torch.zeros_like(x)
            # shape: batch_size, seq_len, seq_len

            # (batch_size, seq_len, [1 seqeezed]), column num for max values of each row
            # shape: (batch_size, seq_len)
            row_largest_idx = torch.argmax(x, dim=2)

            row_largest_mat[torch.arange(batch_size).reshape((-1, 1)), torch.arange(x.size(1)), row_largest_idx] = 1.0

            x = row_largest_mat * row_largest_mat.transpose(1, 2)

            x = x * self.mask_no_sharp_loops
            return x

if __name__ == '__main__':
    pass
