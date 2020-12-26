import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels (filters), 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # linear layers are affine (Dense in keras): y = Wx + b ... bias defaults to True
        # should see whether input size is required
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # max pooling over (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # if input is square, max_pool can take a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions excluding batch
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
# print(net)
# params = list(net.parameters())
# print(len(params))
# print(params[0].size())

input = torch.randn(1, 1, 32, 32)
out = net(input)
# print(out)
# net.zero_grad()
# out.backward(torch.randn(1, 10))

output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss.grad_fn)

net.zero_grad()
loss.backward()

optimizer = optim.SGD(net.parameters(), lr=0.01)
for i in range(10):
    output = net(input)
    loss = criterion(output, target)
    print(loss)
    loss.backward()
    optimizer.step()
