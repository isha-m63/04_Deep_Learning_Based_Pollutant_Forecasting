from models.neural_ode import NeuralODERegressor
from models.registry import get_spec
from training.trainer import fit, main


def train_neural_ode(train_loader, test_loader, input_size, output_size, device, hidden_size, intermediate_size,
                     epochs=50, lr=1e-3, save_path=None, losses_save_path=None):
    model = NeuralODERegressor(input_size=input_size, hidden_size=hidden_size,
                               intermediate_size=intermediate_size, output_size=output_size)
    return fit(get_spec("neural_ode"), model, train_loader, test_loader, device, epochs, lr, save_path, losses_save_path)


if __name__ == '__main__':
    main("neural_ode")
