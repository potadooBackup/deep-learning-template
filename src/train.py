from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb
from data import StockDatasetModule
from models import (BiLSTMModel, GRUModel, LSTMAttentionModel, LSTMModel,
                    RNNModel)

if __name__ == '__main__':

    input_dim = 1
    hidden_dim = 32
    num_layers = 2 
    output_dim = 1

    company_list = ['AAPL', 'COST', 'FB']
    
    for company in company_list:

        rnn = RNNModel(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim, layer_num = num_layers)
        lstm = LSTMModel(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim, layer_num = num_layers)
        gru = GRUModel(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim, layer_num = num_layers)
        bilstm = BiLSTMModel(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim, layer_num = num_layers)
        attnlstm = LSTMAttentionModel(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim, layer_num = num_layers)

        model_dict = {'RNN': rnn, 'LSTM': lstm, 'GRU': gru, 'Bi-LSTM': bilstm, 'Attn-LSTM': attnlstm}

        for name, model in model_dict.items():
            wandb_logger = WandbLogger(project='time series - NASDAQ100_', name = f'{name}-{company}', group = company, tags = [name, company])

            datamodule = StockDatasetModule(company, 10, 1)
            
            trainer = Trainer(logger = wandb_logger, check_val_every_n_epoch=1, max_epochs=50)
            trainer.fit(model = model, datamodule = datamodule)
            trainer.test(model = model, datamodule = datamodule)
            prediction_and_ground_truth = trainer.predict(model = model, datamodule = datamodule)

            prediction = [float(x[0][0]) for x in prediction_and_ground_truth]
            ground_truth = [float(x[1][0]) for x in prediction_and_ground_truth]

            for pred, y in zip(prediction, ground_truth):
                wandb.log({"Predicted Stock Price": pred})
                wandb.log({"Real Stock Price": y})
        
            wandb.finish()