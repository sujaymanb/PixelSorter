from artist import Artist
from artist.random_mover import RandomMoverArtist

class MultiHeadedRandomMoverArtist(Artist):
    def __init__(self, num_heads=2, max_iter=2500, brush_width=30, threshold=75, set_color=False):
        self.num_heads = num_heads
        self.max_iter = max_iter
        self.brush_width = brush_width
        self.threshold = threshold
        self.initialized = False
        self.set_color = set_color
        self.heads = [RandomMoverArtist(max_iter, brush_width, threshold, set_color) for _ in range(num_heads)]
    
    def _init_state(self, img):
        for head in self.heads:
            head._init_state(img)
    
    def step(self, img):
        sorted_img = img.copy()
        for head in self.heads:
            sorted_img = head.step(sorted_img)
        
        return sorted_img
