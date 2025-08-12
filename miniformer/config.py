class Config:

    def __init__(self, **kwargs):

        self._attributes = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        return self._attributes