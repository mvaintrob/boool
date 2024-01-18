from random_network import get_pairwise_and, gen_data
import torch as t

x = get_pairwise_and(t.tensor([[1, 0, 1]]))
print(x)

y = gen_data(3, 0.1, 1)
print(y)