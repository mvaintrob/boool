from typing import Callable, Any, List, Literal, Tuple, Optional
import torch as t
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import sqrt, log
from dataclasses import dataclass
from random import sample

@dataclass
class ExperimentParams:
    """
    All the parameters for the experiment
    The default values should be sensible defaults
    Construct this class and then change any parameters you want to change after construction
    """
    d_0: int = 100
    d_mlp: int = 300
    act_fn: Callable = t.nn.ReLU() # can also use quadratic, relu...
    weight_init_method: Callable = lambda w: t.nn.init.kaiming_normal_
    freeze: bool = True # whether to freeze the projection of the sparse boolean input vectors
    dataset_size: int = 300_000
    batch_size: int = 100
    lr: float = 1e-3
    test_dataset_size: int = 1000
    loss_p: int = 2
    device: t.device = None
    n_plots: Optional[int] = 30
    n_prints: int = 20
    loss_type: Literal["normal", "reweighted"] = "normal"
    operation: Literal["and", "xor"] = "xor"

    # These are computed from the above
    l: float = None
    epsilons: List[float] = None

    def __post_init__(self):
        self.l = 2 * sqrt(log(self.d_0))
        print(f"using E[l]: {self.l}")
        epsilon = log(self.d_mlp) / sqrt(self.d_mlp)
        self.epsilons = [epsilon * x for x in [0.1, 0.2, 0.5, 1]]
        self.device = t.device("cuda:0") if t.cuda.is_available() else t.device("cpu")


class Net(t.nn.Module):
    def __init__(
        self,
        d_0: int,
        d_mlp: int,
        act_fn: Callable,
        weight_init_method: Callable = t.nn.init.kaiming_uniform_,
        freeze_first_layer: bool = True,
    ):
        super().__init__()
        self.linear1 = t.nn.Linear(d_0, d_mlp)
        self.linear2 = t.nn.Linear(d_mlp, (d_0 * (d_0 - 1)) // 2)
        self.act_fn = act_fn
        self.freeze_first_layer = freeze_first_layer

        weight_init_method(self.linear1.weight)
        weight_init_method(self.linear2.weight)

        if freeze_first_layer:
            self.linear1.weight.requires_grad = False
            self.linear1.bias.requires_grad = False

    def forward(self, x) -> Any:
        return self.linear2(self.act_fn(self.linear1(x)))


def get_param_mean_var(net: Net) -> Tuple[float]:
    linear_1_params = (
        t.nn.utils.parameters_to_vector(net.linear1.parameters())
        .detach()
        .cpu()
        .flatten()
    )
    return linear_1_params.mean().item(), linear_1_params.std().item()


def get_pairwise_and(x: t.Tensor) -> t.Tensor:
    """
    x is a batch of sparse d_0-dim boolean vectors
    """
    all_ands = t.einsum("bi,bj->bij", x, x)
    rows, cols = t.triu_indices(x.shape[1], x.shape[1], offset=1)
    return all_ands[:, rows, cols].reshape(x.shape[0], -1)

def get_pairwise_xor(x: t.Tensor) -> t.Tensor:
    """
    x is a batch of sparse d_0-dim boolean vectors
    """
    all_xors = t.logical_xor(x.unsqueeze(2), x.unsqueeze(1)).float()
    rows, cols = t.triu_indices(x.shape[1], x.shape[1], offset=1)
    return all_xors[:, rows, cols].reshape(x.shape[0], -1)

def get_expected(x: t.Tensor, operation: Literal["and", "xor"]) -> t.Tensor:
    if operation == "and":
        return get_pairwise_and(x)
    elif operation == "xor":
        return get_pairwise_xor(x)
    else:
        raise ValueError(f"Unknown operation: {operation}")

def u_op_loss(
    x: t.Tensor, output: t.Tensor, operation: Literal["and", "xor"], p: int = 6, device: t.device = t.device("cpu")
) -> t.Tensor:
    """
    x is a batch of sparse d_0-dim boolean vectors
    output is the predicted pairwise ands of all the entries of each vector (batch_size x d choose 2)
    """
    expected = get_expected(x.cpu(), operation).to(device)
    return (output - expected).abs().pow(p).mean() ** (1 / p)


def u_and_loss_reweighted(
    x: t.Tensor,
    output: t.Tensor,
    operation: Literal["and", "xor"],
    p: int = 6,
    ones_weighting: float = 1.0,
    device: t.device = t.device("cpu"),
) -> t.Tensor:
    """
    x is a batch of sparse d_0-dim boolean vectors
    output is the predicted pairwise ands of all the entries of each vector (batch_size x d choose 2)
    here we reweight the loss so that the loss on the ones is weighted by ones_weighting
    """
    expected = get_expected(x.cpu(), operation).to(device)
    ones_mask = expected == 1
    zeros_mask = expected == 0
    n_ones = ones_mask.sum()
    n_zeros = zeros_mask.sum()
    ones_loss = ((output[ones_mask] - 1).abs().pow(p).sum() / n_ones) ** (1 / p)
    zeros_loss = ((output[zeros_mask]).abs().pow(p).sum() / n_zeros) ** (1 / p)
    loss = ones_weighting * ones_loss + zeros_loss
    if loss < 0:
        raise ValueError(
            f"Loss is negative: {loss}, ones_loss: {ones_loss}, zeros_loss: {zeros_loss}, expected.sum() {expected.sum()}"
        )
    return loss


def gen_data(d_0: int, p_on: float, dataset_size: int) -> t.Tensor:
    """
    Generates a batch of sparse d-dim boolean vectors
    """
    return (t.rand(dataset_size, d_0) < p_on).float()


def test_accuracy(
    net: Net,
    dataset_size: int,
    l: float,
    epsilons: List[float],
    operation: Literal["and", "xor"],
    do_plot: bool = False,
    idx: int = 0,
    device: t.device = t.device("cpu"),
    n_hist_bins: int = 100,
) -> float:
    p_on = l / net.linear1.in_features
    data = gen_data(net.linear1.in_features, p_on, dataset_size)
    data = data.to(device)
    output = net(data)
    expected = get_expected(data.cpu(), operation).to(device)

    mask = expected == 0
    n_zeros = mask.sum()
    means_at_zeros = (output * mask).sum() / n_zeros
    std_at_zeros = (((output - means_at_zeros) * mask).pow(2).sum() / n_zeros).sqrt()

    mask = expected == 1
    n_ones = mask.sum()
    means_at_ones = (output * mask).sum() / n_ones
    std_at_ones = (((output - means_at_ones) * mask).pow(2).sum() / n_ones).sqrt()

    all_errors = (output - expected).abs().flatten()

    one_errors = all_errors[expected.flatten() == 1].detach().cpu().numpy().tolist()
    zero_errors = all_errors[expected.flatten() == 0].detach().cpu().numpy().tolist()

    if do_plot:
        # Save histogram plot of errors
        plt.hist(one_errors, bins=n_hist_bins, alpha=0.5, label="ones", color="orange")
        plt.hist(sample(zero_errors, k=len(one_errors)), bins=n_hist_bins, alpha=0.5, label="zeros", color="blue")
        plt.xlim([0, 1.3])
        plt.legend()
        plt.title(f"errors for {operation}")
        plt.savefig(f"hist_{idx}.png")
        plt.close()

    return {
        "mean_at_zeros": means_at_zeros.item(),
        "std_at_zeros": std_at_zeros.item(),
        "mean_at_ones": means_at_ones.item(),
        "std_at_ones": std_at_ones.item(),
        "mean_errors": all_errors.mean().item(),
        "std_errors": all_errors.std().item(),
        **{
            f"{round(epsilon, 4)}_acc": all_errors.lt(epsilon).sum().item()
            / output.numel()
            for epsilon in epsilons
        },
    }


def train(
    net: Net,
    dataset_size: int,
    batch_size: int,
    l: float,
    lr: float,
    test_dataset_size: int,
    epsilons: List[float],
    n_plots: Optional[int],
    n_prints: int,
    device: t.device,
    loss_p: int,
    loss_type: Literal["normal", "reweighted"],
    operation: Literal["and", "xor"],
):
    """
    net: the network to train
    dataset_size: how many random sparse boolean examples to train on
    batch_size: batch size to use for training
    l: expected number of ones in each sparse boolean vector
    lr: learning rate for SGD
    test_dataset_size: how many random sparse boolean examples to use for testing
    epsilons: list of epsilons to track for accuracy calculation
    n_plots: how many histogram plots to make of the error size during training
    n_prints: how many times to print the loss and accuracy during training
    device: device to use for training
    loss_p: p to use for the loss function
    operation: which operation to use for the loss function
    """
    mean, var = get_param_mean_var(net)
    print(f"linear1 weights - mean: {mean}, var: {var}")
    net.train()
    optim = t.optim.Adam(net.parameters(), lr=lr)
    p_on = l / net.linear1.in_features
    data = gen_data(net.linear1.in_features, p_on, dataset_size)
    data = data.to(device)
    if n_plots is None:
        plot_every = None
    else:
        plot_every = (dataset_size / batch_size) // n_plots
    print_every = (dataset_size / batch_size) // n_prints
    for batch_idx in tqdm(range(0, dataset_size // batch_size)):
        i = batch_idx * batch_size
        batch = data[i : i + batch_size].to(device)
        optim.zero_grad()
        if loss_type == "normal":
            loss = u_op_loss(x=batch, output=net(batch), operation=operation, device=device, p=loss_p)
        elif loss_type == "reweighted":
            loss = u_and_loss_reweighted(x=batch, output=net(batch), operation=operation, device=device, p=loss_p)
        if batch_idx % print_every == 0:
            print(f"loss: {loss.item()}")
            print(test_accuracy(net=net, dataset_size=test_dataset_size, l=l, epsilons=epsilons, operation=operation, device=device))
        if plot_every is not None:
            if batch_idx % plot_every == 0:
                test_accuracy(net=net, dataset_size=test_dataset_size, l=l, epsilons=epsilons, do_plot=True, idx=i, operation=operation, device=device)
        loss.backward()
        optim.step()
    print(test_accuracy(net=net, dataset_size=test_dataset_size, l=l, epsilons=epsilons, operation=operation, device=device))
    mean, var = get_param_mean_var(net)
    print(f"linear1 weights - mean: {mean}, var: {var}")


def experiment(params: ExperimentParams):
    net = Net(
        d_0=params.d_0,
        d_mlp=params.d_mlp,
        act_fn=params.act_fn,
        weight_init_method=params.weight_init_method,
        freeze_first_layer=params.freeze,
    )
    net = net.to(params.device)
    train(
        net=net,
        dataset_size=params.dataset_size,
        batch_size=params.batch_size,
        l=params.l,
        lr=params.lr,
        test_dataset_size=params.test_dataset_size,
        epsilons=params.epsilons,
        n_plots=params.n_plots,
        n_prints=params.n_prints,
        device=params.device,
        loss_p=params.loss_p,
        loss_type=params.loss_type,
        operation=params.operation,
    )


if __name__ == "__main__":
    params = ExperimentParams()
    experiment(params)