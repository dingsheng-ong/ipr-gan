
class StatsAccumulator(object):
    def __init__(self, ):
        self.reset()

    def update(self, x, n=1):
        self.x += x * n
        self.n += n

    @property
    def sum(self):
        return self.x

    @property
    def avg(self):
        return self.x / (1 if not self.n else self.n)

    @property
    def count(self):
        return self.n

    def reset(self):
        self.x = 0
        self.n = 0
    