from typing import List

import torch
import numpy as np


def print_unititialised_matrix(dim: List):
    x = torch.empty(*dim)
    print(x)


def print_random_matrix(dim: List):
    x = torch.rand(*dim)
    print(x)


def print_zeros(dim: List):
    x = torch.zeros(*dim)
    print(x)


def print_tensor_from_data(data: List):
    x = torch.tensor(data)
    print(x)


def print_randn_from_previous(previous_tensor):
    x = torch.randn_like(previous_tensor, dtype=torch.float)
    print(x)


def main():
    print("Tensors:")

    print_unititialised_matrix([5, 3])
    print_random_matrix([5, 3])
    print_zeros([5, 3])
    print_tensor_from_data([[5.0, 3.0], [0, 1.0]])
    prev = torch.tensor([[5.0, 3.0], [0, 1.0]])
    print_randn_from_previous(prev)

    print(f"\n{'-'*8} OPERATIONS {'-'*8}")
    print("Addition:")
    x = torch.rand(5, 3)
    y = torch.rand(5, 3)
    assert x.shape == y.shape
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"x+y: {x+y}")

    print("Addition with result as arg")
    result = torch.empty(*x.shape)
    print(result)
    torch.add(x, y, out=result)
    print(f"x+y: {result}")

    print("Addition inplace on y")
    y.add_(x)
    print(f"x+y: {y}")

    print("Any operations mutating a tensor in-place are suffixed with an underscore '_'")

    print("Examples of slicing (numpy notation) and resizing (tensor.view)")
    x = torch.randn(4, 4)
    y = x.view(16)
    z = x.view(-1, 8)
    print("x:", x)
    print("x[2:]:", x[2:])
    print(x.size(), y.size(), z.size())

    print("-"*8, "torch -> numpy bridge:", "-"*8)
    a = torch.ones(5)
    b = a.numpy()
    print(a)
    print(b)
    a.add_(1)
    print(a)
    print(b)

    print("-" * 8, "numpy -> torch bridge:", "-" * 8)
    a = np.ones(5)
    b = torch.from_numpy(a)
    print(a)
    print(b)
    np.add(a, 1, out=a)
    print(a)
    print(b)

    if torch.cuda.is_available():
        device = torch.device("cuda")  # a CUDA device object
        y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
        x = x.to(device)  # or just use strings ``.to("cuda")``
        z = x + y
        print(z)
        print(z.to("cpu", torch.double))  # ``.to`` can also change dtype together!


if __name__ == "__main__":
    main()
