from models.lstm import LSTMRegressor
from models.registry import get_spec
from training.trainer import fit, main


def train_lstm(train_loader, test_loader, input_size, output_size, device, hidden_size, num_layers, dropout,
               epochs=50, lr=1e-3, save_path=None, losses_save_path=None):
    model = LSTMRegressor(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          output_size=output_size, dropout=dropout)
    return fit(get_spec("lstm"), model, train_loader, test_loader, device, epochs, lr, save_path, losses_save_path)


if __name__ == '__main__':
    main("lstm")
