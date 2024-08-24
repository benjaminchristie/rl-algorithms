import numpy as np
from scipy.stats import halfnorm


class ReplayMemory(object):

    def __init__(self, capacity=1000):
        self.capacity = int(capacity)
        self.position = 0
        self.size = 0
        self.buffer = np.zeros(self.capacity, dtype=tuple)

    def push(self, *args):
        self.buffer[self.position] = args
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        batch = np.random.choice(self.buffer[0 : self.size], batch_size)
        args = map(np.stack, zip(*batch))
        return args

    def weighted_sample(self, batch_size, stdev=10.0):
        weights = np.array(
            halfnorm.pdf(np.arange(0, self.capacity), loc=0, scale=stdev)
        )
        weights = weights.take(np.arange(self.position - self.capacity, self.position))[
            ::-1
        ][0 : self.size]
        weights /= np.sum(weights)
        batch = np.random.choice(self.buffer[0 : self.size], batch_size, p=weights)
        args = map(np.stack, zip(*batch))
        return args

    def __len__(self):
        return self.size


class ReplayMemoryRestricted(ReplayMemory):

    def __init__(self, capacity=1000, n_types=5):
        super().__init__(capacity=capacity)
        self.n_types = n_types

    def push(self, *args):
        assert len(args) == self.n_types
        super().push(args)
