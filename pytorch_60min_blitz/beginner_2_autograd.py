import torch

if __name__ == "__main__":
    x = torch.ones(size=[2, 2], requires_grad=True)
    y = x + 2
    z = y * y * 3
    out = z.mean()
    print(z, out)
    out.backward()
    print(x.grad)
