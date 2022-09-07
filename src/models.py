import math
from abc import abstractmethod

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.autograd import Variable
from torch.nn import MSELoss
from torch.nn import functional as F
from torch.optim import Adam
from torchmetrics import (MeanAbsoluteError, MeanAbsolutePercentageError,
                          MeanSquaredError)
from torchmetrics.functional import accuracy


class BaseModel(LightningModule):

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, cell, 
                 lr = 1e-3, ):
        '''method used to define our model parameters'''
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.loss = MSELoss()
        self.lr = lr

        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        if cell == "RNN":
            self.cell = nn.RNN(input_size=self.input_dim, hidden_size=self.hidden_dim,
                               num_layers=self.layer_num, dropout=0.0,
                               nonlinearity="tanh", batch_first=True,)
        if cell == "LSTM":
            self.cell = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                                num_layers=self.layer_num, dropout=0.0,
                                batch_first=True, )
        if cell == "GRU":
            self.cell = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim,
                               num_layers=self.layer_num, dropout=0.0,
                               batch_first=True, )
        if cell == "BiLSTM":
            self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)
            self.cell = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                                num_layers=self.layer_num, dropout=0.0,
                                batch_first=True, bidirectional=True)

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        _, loss = self._get_preds_loss(batch)

        # Log loss and metric
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        preds, loss = self._get_preds_loss(batch)

        # Log loss and metric
        self.log('val_loss', loss)
        # Let's return preds to use it in a custom callback
        return preds

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''

        x, y = batch
        preds = self(x)

        mse = MeanSquaredError().to(self.device)
        mae = MeanAbsoluteError().to(self.device)
        mape = MeanAbsolutePercentageError().to(self.device)

        test_loss = mse(preds, y)
        rmse_loss = math.sqrt(test_loss)
        mae_loss = mae(preds, y)
        mape_loss = mape(preds, y)

        # Log loss and metric
        self.log('test_loss', test_loss)
        self.log('rmse_loss', rmse_loss)
        self.log('mae_loss', mae_loss)
        self.log('mape_loss', mape_loss)
    
    def predict_step(self, batch, batch_idx):

        x, y = batch
        preds = self(x)

        return preds, y

    def configure_optimizers(self):
        '''defines model optimizer'''
        return Adam(self.parameters(), lr=self.lr)
    
    def _get_preds_loss(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        return preds, loss

class RNNModel(BaseModel):

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, cell = 'RNN'):
        super().__init__(input_dim, hidden_dim, output_dim, layer_num, cell)

    def forward(self, x):

        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.layer_num * 1, batch_size , self.hidden_dim).to(self.device))
        rnn_output, hn = self.cell(x, h0)
        fc_output = self.fc(rnn_output[:, -1, :])

        return fc_output

class LSTMModel(BaseModel):

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, cell = 'LSTM'):
        super().__init__(input_dim, hidden_dim, output_dim, layer_num, cell)

    def forward(self, x):

        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.layer_num * 1, batch_size, self.hidden_dim).to(self.device))
        c0 = Variable(torch.zeros(self.layer_num * 1, batch_size, self.hidden_dim).to(self.device))
        rnn_output, (hn, cn) = self.cell(x, (h0, c0))
        fc_output = self.fc(rnn_output[:, -1, :])

        return fc_output

class GRUModel(BaseModel):

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, cell = 'GRU'):
        super(GRUModel, self).__init__(input_dim, hidden_dim, output_dim, layer_num, cell)

    def forward(self, x):

        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.layer_num * 1, batch_size, self.hidden_dim).to(self.device))
        rnn_output, hn = self.cell(x, h0)
        fc_output = self.fc(rnn_output[:, -1, :])

        return fc_output

class BiLSTMModel(BaseModel):

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, cell = 'BiLSTM'):
        super().__init__(input_dim, hidden_dim, output_dim, layer_num, cell)

    def forward(self, x):

        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.layer_num * 2, batch_size, self.hidden_dim).to(self.device))
        c0 = Variable(torch.zeros(self.layer_num * 2, batch_size, self.hidden_dim).to(self.device))
        rnn_output, (hn, cn) = self.cell(x, (h0, c0))
        fc_output = self.fc(rnn_output[:, -1, :])

        return fc_output

class LSTMAttentionModel(BaseModel):

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, cell = 'LSTM'):
        super().__init__(input_dim, hidden_dim, output_dim, layer_num, cell)

    def attention(self, rnn_output: torch.Tensor, hn: torch.Tensor):
        hn = hn.unsqueeze(0) # [1 * batch_size * hidden_dim]
        hn = hn.permute(1, 2, 0)

        # rnn_output: [batch_size * seq_len * hidden_dim]
        attn_weights = torch.bmm(rnn_output, hn) # attention_weighs: [batch_size * seq_len * 1]
        attn_weights = attn_weights.permute(1, 0, 2).squeeze(-1)
        attention = F.softmax(attn_weights, 1)
        attn_out = torch.bmm(rnn_output.transpose(1, 2), attention.unsqueeze(-1).transpose(1,0))
        return attn_out.squeeze()

    def forward(self, x):

        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.layer_num * 1, batch_size, self.hidden_dim).to(self.device))
        c0 = Variable(torch.zeros(self.layer_num * 1, batch_size, self.hidden_dim).to(self.device))
        rnn_output, (hn, cn) = self.cell(x, (h0, c0))
        attn_out = self.attention(rnn_output, hn[-1])
        fc_output = self.fc(attn_out)

        return fc_output