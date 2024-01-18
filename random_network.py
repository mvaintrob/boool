from typing import Callable, Any, List
import torch as t
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import sqrt, log

class Net(t.nn.Module):

    def __init__(self, d_0, d_mlp, act_fn: Callable, weight_init_method: Callable = t.nn.init.kaiming_uniform_, freeze_first_layer: bool = True):
        super().__init__()
        self.linear1 = t.nn.Linear(d_0, d_mlp)
        self.linear2 = t.nn.Linear(d_mlp, (d_0 * (d_0 -1))//2)
        self.act_fn = act_fn
        self.freeze_first_layer = freeze_first_layer

        weight_init_method(self.linear1.weight)
        weight_init_method(self.linear2.weight)

        if freeze_first_layer:
            self.linear1.weight.requires_grad = False
            self.linear1.bias.requires_grad = False

    def forward(self, x) -> Any:
        return self.linear2(self.act_fn(self.linear1(x)))


def get_pairwise_and(x):
    """
    x is a batch of sparse d-dim boolean vectors
    """
    all_ands = t.einsum('bi,bj->bij', x, x)
    rows, cols = t.triu_indices(x.shape[1], x.shape[1], offset=1)
    return all_ands[:, rows, cols].reshape(x.shape[0], -1)

def u_and_loss(x, output, p=6, device: t.device = t.device("cpu")) -> float:
    """
    x is a batch of sparse d-dim boolean vectors
    output is the pairwise ands of all the entries of each vector (batch times d choose 2 dim)
    """
    expected = get_pairwise_and(x.cpu()).to(device)
    return (output - expected).abs().pow(p).mean() ** (1/p)

def u_and_loss_reweighted(x, output, p=6, ones_weighting: float = 1.0, device: t.device = t.device("cpu")) -> float:
    """
    x is a batch of sparse d-dim boolean vectors
    output is the pairwise ands of all the entries of each vector (batch times d choose 2 dim)
    """
    expected = get_pairwise_and(x.cpu()).to(device)
    ones_loss = ((output[expected==1] - 1).abs().pow(p).sum() / (expected==1).sum()) ** (1/p)
    zeros_loss = ((output[expected==0]).abs().pow(p).sum() / (expected==0).sum()) ** (1/p)
    loss = ones_weighting * ones_loss + zeros_loss
    if loss < 0:
        raise ValueError(f"Loss is negative: {loss}, ones_loss: {ones_loss}, zeros_loss: {zeros_loss}, expected sum {expected.sum()}")
    return loss

def gen_data(d_0: int, p_on: float, dataset_size: int):
    """
    Generates a batch of sparse d-dim boolean vectors
    """
    return (t.rand(dataset_size, d_0) < p_on).float()

def test_accuracy(net, dataset_size, l: float, epsilons: List[float], do_plot: bool = False, idx: int = 0, device: t.device = t.device("cpu")) -> float:
    p_on = l / net.linear1.in_features
    data = gen_data(net.linear1.in_features, p_on, dataset_size)
    data = data.to(device)
    output = net(data)
    expected = get_pairwise_and(data.cpu()).to(device)

    mask = expected == 0
    mask_ones = output * mask 
    means_at_zeros = mask_ones.sum() / mask.sum()
    std_at_zeros = ((mask_ones - means_at_zeros) * mask).pow(2).sum().sqrt() / mask.sum().sqrt()

    mask = expected == 1
    mask_zeros = output * mask
    means_at_ones = mask_zeros.sum() / mask.sum()
    std_at_ones = ((mask_zeros - means_at_ones) * mask).pow(2).sum().sqrt() / mask.sum().sqrt()

    all_errors = (output - expected).abs().flatten()
    # Save histogram plot of errors
    if do_plot:
        plt.hist(all_errors.detach().cpu().numpy(), bins=100)
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
            f"{round(epsilon, 4)}_acc":  all_errors.lt(epsilon).sum().item() / output.numel() for epsilon in epsilons
        }
    }
    
def train(net, dataset_size, batch_size, l: float, lr=1e-3, test_dataset_size: int = 100, epsilons: List[float] = [0.1], n_plots = 10, device: t.device = t.device("cpu"), loss_p: int = 6):
    net.train()
    optim = t.optim.Adam(net.parameters(), lr=lr)
    p_on = l / net.linear1.in_features
    data = gen_data(net.linear1.in_features, p_on, dataset_size)
    data = data.to(device)

    plot_every = (dataset_size / batch_size) // n_plots
    for batch_idx in tqdm(range(0, dataset_size//batch_size)):
        i = batch_idx * batch_size
        batch = data[i:i+batch_size].to(device)
        optim.zero_grad()
        loss = u_and_loss_reweighted(batch, net(batch), device=device, p=loss_p)
        if batch_idx % 300 == 0:
            print(f"loss: {loss.item()}")
            print(test_accuracy(net, test_dataset_size, l, epsilons, device=device))
        # if batch_idx % plot_every == 0:
        #     test_accuracy(net, test_dataset_size, l, epsilons, do_plot=True, idx=i, device=device)
        loss.backward()
        optim.step()

    print(test_accuracy(net, test_dataset_size, l, epsilons, device=device))



if __name__ == "__main__":
    d_0 = 100
    d_mlp = 300
    act_fn = t.nn.ReLU()
    weight_init_method = t.nn.init.kaiming_normal_
    freeze = False
    net = Net(d_0, d_mlp, act_fn, weight_init_method, freeze_first_layer=freeze)
    dataset_size = 1_000_000
    batch_size = 100
    # l = sqrt(log(d_0))
    l = 4
    lr=1e-3
    test_dataset_size=1000
    epsilon = log(d_mlp) / sqrt(d_mlp)
    epsilons = [epsilon*x for x in [0.1, 0.3, 1]]
    loss_p = 7
    device = t.device("cuda:0") if t.cuda.is_available() else t.device("cpu")
    net = net.to(device)
    train(net, dataset_size, batch_size, l, lr, test_dataset_size, epsilons, device=device, loss_p=loss_p)



    
    
    


