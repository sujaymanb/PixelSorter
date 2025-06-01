from abc import ABC, abstractmethod

class Artist(ABC):
    @abstractmethod
    def step(self, img):
        """Perform one iteration of the artist's effect on the image."""
        pass