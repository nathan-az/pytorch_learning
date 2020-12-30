import torch


class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()
        self.net = MarioNet(self.state_dim, self.action_dim).float()

    def act(self, state):
        pass

    def cache(self, experience):
        pass

    def recall(self):
        pass

    def learn(self):
        pass
