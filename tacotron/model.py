import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from torchaudio import transforms
from torch import nn


class BatchNormConvolution(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            stride=1,
            activation=nn.ReLU
    ):
        super().__init__()
        self.batchnorm = nn.Sequential(
            nn.ConstantPad1d(padding, 0),
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=False),
            activation(),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        return self.batchnorm(x)


class EncoderCBHD(nn.Module):
    def __init__(self,
                 feature_sizes=128,
                 K=16
    ):
        super().__init__()
        self.convolve_sets = nn.ModuleList([BatchNormConvolution(
            in_channels=feature_sizes,
            out_channels=feature_sizes,
            kernel_size=x,
            padding=[(x-1)//2, x//2]
        ) for x in range(1, K+1)])
        self.pre_pool_padding = nn.ConstantPad1d((1, 0), 0)
        self.pooling = nn.MaxPool1d(2, 1, 0)

        self.conv1dprojections = nn.ModuleList([
            BatchNormConvolution(
            in_channels= K*feature_sizes,
            out_channels=feature_sizes,
            kernel_size=3,
            padding=[1, 1]),
            BatchNormConvolution(
            in_channels= feature_sizes,
            out_channels=feature_sizes,
            kernel_size=3,
            padding=[1, 1])
        ])
        self.highway_net = nn.Sequential(
            nn.Linear(feature_sizes, feature_sizes),
            nn.ReLU(),
            nn.Linear(feature_sizes, feature_sizes),
            nn.ReLU(),
            nn.Linear(feature_sizes, feature_sizes),
            nn.ReLU(),
            nn.Linear(feature_sizes, feature_sizes),
            nn.ReLU()
        )
        self.gru = nn.GRU(input_size=feature_sizes,
                          hidden_size=feature_sizes,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True
        )

    def forward(self, input):
        convs = []
        for conv in self.convolve_sets:
            convs.append(conv(input))
        x = torch.cat(convs, dim=1)
        x = self.pre_pool_padding(x)
        x = self.pooling(x)
        for conv in self.conv1dprojections:
            x = conv(x)
        x = input+x
        x = x.transpose(1,2)
        x = self.highway_net(x)
        x = self.gru(x)[0]
        return x


class TacotronEncoder(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.char_embedding = nn.Embedding(self.input_shape, 256)
        self.encoder_prenet = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.Dropout(0.5)
        )
        self.CBHD = EncoderCBHD(128)

    def forward(self, x):
        x = self.char_embedding(x)
        x = self.encoder_prenet(x)
        cbhd_result = self.CBHD(x.transpose(1,2))

        return cbhd_result


class TacotronAttention(nn.Module):
    def __init__(self, input_dim, work_dim, reduction_factor, num_mel):
        super().__init__()
        self.input_dim = input_dim
        self.work_dim = work_dim
        self.num_mel = num_mel
        self.reduction_factor = reduction_factor
        self.decoder_rnn = nn.GRU(self.input_dim, self.work_dim, batch_first=True)
        self.decoder_linear = nn.Linear(self.work_dim, self.work_dim)
        self.encoder_linear = nn.Linear(self.work_dim, self.work_dim)
        self.alignment_scores_linear = nn.Linear(self.work_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.before_rnn_projection = nn.Linear(2*work_dim, work_dim)
        self.gru1 = nn.GRU(self.work_dim, self.work_dim, batch_first=True)
        self.gru2 = nn.GRU(self.work_dim, self.work_dim, batch_first=True)
        self.output_linear = nn.Linear(self.work_dim, self.num_mel * self.reduction_factor)

    def forward(self, decoder_output, encoder_output, prev_hidden_state = None, prev_gru1=None, prev_gru2=None):
        decoder_rnn_output, new_attention_hidden_states = self.decoder_rnn(decoder_output, prev_hidden_state)
        decoder_output = self.decoder_linear(decoder_rnn_output) ### [batch_sze, 1, hidden_state]
        encoder_output = self.encoder_linear(encoder_output) ### [batch_sze, time, hidden_state]
        attention_weights = decoder_output + encoder_output ### [batch_sze, time, hidden_state]
        attention_weights = self.alignment_scores_linear(attention_weights) ### [batch_sze, time, 1]
        attention_weights = self.softmax(attention_weights) ### [batch_sze, time, 1]
        context_vector = (encoder_output.transpose(1,2) @ attention_weights).transpose(1,2)
        context_vector = torch.cat([decoder_output, context_vector], dim=2)
        context_vector = self.before_rnn_projection(context_vector)

        out_gru1, hidden_gru1 = self.gru1(context_vector, prev_gru1)
        out_gru1 = out_gru1 + context_vector
        out_gru2, hidden_gru2 = self.gru2(out_gru1, prev_gru2)
        out_gru2 = out_gru2 + out_gru1
        output = self.output_linear(out_gru2)
        output = output.view(-1, self.reduction_factor, self.num_mel)

        return output, new_attention_hidden_states, hidden_gru1, hidden_gru2


class TacotronDecoder(nn.Module):
    def __init__(self, n_mels, hidden_dim, reduction_factor, max_iter=500):
        super().__init__()
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.reduction_factor = reduction_factor
        self.decoder_prenet = nn.Sequential(
            nn.Linear(self.n_mels, 256),
            nn.Dropout(0.5),
            nn.Linear(256, self.hidden_dim),
            nn.Dropout(0.5)
        )
        self.max_iter = max_iter
        self.attention = TacotronAttention(self.hidden_dim, 256, self.reduction_factor, self.n_mels)

    def forward(self, x, encoder_vec):
        x = self.decoder_prenet(x)
        sequence_len = x.shape[1] // self.reduction_factor
        final_len = x.shape[1]
        mels = []
        att_hs = None
        gru1_hs = None
        gru2_hs = None
        attention_input = x[:, self.reduction_factor-1:self.reduction_factor, :]
        for iter in range(1, sequence_len+2):
            output, att_hs, gru1_hs, gru2_hs = self.attention(attention_input, encoder_vec, att_hs, gru1_hs, gru2_hs)
            mels.append(output)
            attention_input = x[:, iter*self.reduction_factor-1:iter*self.reduction_factor, :]

        mels = torch.cat(mels, dim=1)[:, :final_len, :]

        return mels

    def predict_mel(self, x, encoder_vec):
        x = self.decoder_prenet(x)
        att_hs = None
        gru1_hs = None
        gru2_hs = None
        attention_input = x
        mels = []
        for iter in range(1, self.max_iter):
            output, att_hs, gru1_hs, gru2_hs = self.attention(attention_input, encoder_vec, att_hs, gru1_hs, gru2_hs)
            mels.append(output)
            attention_input = self.decoder_prenet(output[:, -1:, :])

        mels = torch.cat(mels, dim=1)
        return mels


class PPCBHD(nn.Module):
    def __init__(self,
                 feature_sizes=80,
                 hidden_dim=128,
                 K=8
    ):
        super().__init__()
        self.convolve_sets = nn.ModuleList([BatchNormConvolution(
            in_channels=feature_sizes,
            out_channels=hidden_dim,
            kernel_size=x,
            padding=[(x-1)//2, x//2]
        ) for x in range(1, K+1)])
        self.pre_pool_padding = nn.ConstantPad1d((1, 0), 0)
        self.pooling = nn.MaxPool1d(2, 1, 0)

        self.conv1dprojections = nn.ModuleList([
            BatchNormConvolution(
            in_channels= K*hidden_dim,
            out_channels=2*hidden_dim,
            kernel_size=3,
            padding=[1, 1]),
            BatchNormConvolution(
            in_channels= 2*hidden_dim,
            out_channels=feature_sizes,
            kernel_size=3,
            padding=[1, 1])
        ])
        self.highway_net = nn.Sequential(
            nn.Linear(feature_sizes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.gru = nn.GRU(input_size=hidden_dim,
                          hidden_size=hidden_dim,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True
        )

    def forward(self, input):
        convs = []
        for conv in self.convolve_sets:
            convs.append(conv(input))
        x = torch.cat(convs, dim=1)
        x = self.pre_pool_padding(x)
        x = self.pooling(x)
        for conv in self.conv1dprojections:
            x = conv(x)
        x = input + x
        x = x.transpose(1, 2)
        x = self.highway_net(x)
        x = self.gru(x)[0]
        return x


class PostProcess(nn.Module):
    def __init__(self, final_shape=1024):
        super().__init__()
        self.final_shape = final_shape+1
        self.CBHD = PPCBHD(80, 128, 8)
        self.final_layer = nn.Linear(256, self.final_shape)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.CBHD(x)
        x = self.final_layer(x)
        x = x.transpose(1, 2)
        return x


class Tacotron(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = TacotronEncoder(self.vocab_size)
        self.decoder = TacotronDecoder(80, 128, 5)
        self.post_processer = PostProcess()

    def forward(self, texts, mels):
        encoder_output = self.encoder(texts)
        mels = mels.transpose(1, 2)
        decoder_output = self.decoder(mels, encoder_output)
        pp_output = self.post_processer(decoder_output)
        return decoder_output.transpose(1, 2), pp_output.transpose(1, 2)

    def predict(self, texts):
        encoder_output = self.encoder(texts)
        zero_mels = torch.zeros(texts.shape[0], 1, 80)
        predict = self.decoder.predict_mel(zero_mels, encoder_output)
        predict = self.post_processer(predict)
        return predict

