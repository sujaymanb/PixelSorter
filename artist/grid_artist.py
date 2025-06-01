from artist import Artist

class GridArtist(Artist):
    def __init__(self, rows=32, cols=16, direction='horizontal'):
        self.rows = rows
        self.cols = cols
        self.i = 0
        self.j = 0
        self.initialized = True
        if direction == 'horizontal':
            self.axis = 0
        elif direction == 'vertical':
            self.axis = 1


    def step(self, img):
        # Process one region per step
        if self.i >= self.rows:
            return img  # Done

        x_start = self.i * (img.shape[0] // self.rows)
        x_end = (self.i + 1) * (img.shape[0] // self.rows)
        y_start = self.j * (img.shape[1] // self.cols)
        y_end = (self.j + 1) * (img.shape[1] // self.cols)

        region = img[x_start:x_end, y_start:y_end]
        sorted_region = np.sort(region, axis=self.axis)
        img[x_start:x_end, y_start:y_end] = sorted_region

        # Move to next region
        self.j += 1
        if self.j >= self.cols:
            self.j = 0
            self.i += 1

        return img