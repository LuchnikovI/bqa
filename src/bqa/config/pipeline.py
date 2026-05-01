class Pipe:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwds):
        return self.func(*args, **kwds)

    def __or__(self, other):
        return Pipe(lambda x: other(self(x)))

    def __ror__(self, value):
        return self(value)

def pipeline(func):
    return Pipe(func)
