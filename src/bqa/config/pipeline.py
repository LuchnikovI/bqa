class pipeline:
    def __init__(self, func):
        self.__doc__ = func.__doc__
        self.func = func

    def __call__(self, *args, **kwds):
        return self.func(*args, **kwds)

    def __or__(self, other):
        return pipeline(lambda x: other(self(x)))

    def __ror__(self, value):
        return self(value)

