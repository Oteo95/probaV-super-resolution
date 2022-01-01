from abc import ABCMeta, abstractmethod


class SrganBase(metaclass=ABCMeta):

    def __init__(self, **kwargs):
        pass

    def __repr__(self):
        return f"SRGAN"

    @abstractmethod
    def generator(self):
        pass

    @abstractmethod
    def discriminator(self):
        pass
    
    @abstractmethod
    def srgan(self, generator, discriminator):
        pass
