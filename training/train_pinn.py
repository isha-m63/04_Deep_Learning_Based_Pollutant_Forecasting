from models.pinn import PINNRegressor
from models.registry import get_spec
from training.trainer import fit, main


def train_pinn(train_loader, test_loader, input_size, output_size, num_layers, device, hidden_size,
               epochs=50, lr=1e-3, save_path=None, losses_save_path=None, lambda_phys=1.0, delta_t=1.0):
    """MLP with a temporal-smoothness penalty: lambda_phys * mean(((f(x_t) - f(x_{t-1})) / delta_t)^2)."""
    spec = get_spec("pinn")
    model = PINNRegressor(input_size, hidden_size, output_size, num_layers)
    extra = spec.make_extra_loss({"pinn_lambda_phys": lambda_phys, "pinn_delta_t": delta_t})
    return fit(spec, model, train_loader, test_loader, device, epochs, lr, save_path, losses_save_path, extra_loss=extra)


if __name__ == '__main__':
    main("pinn")
