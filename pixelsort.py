import cv2
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
import tyro

class Artist(ABC):
    @abstractmethod
    def step(self, img):
        """Perform one iteration of the artist's effect on the image."""
        pass

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

class RandomMoverArtist(Artist):
    def __init__(self, max_iter=2500, brush_width=30, threshold=75, set_color=False):
        self.max_iter = max_iter
        self.brush_width = brush_width
        self.threshold = threshold
        self.initialized = False
        self.set_color = set_color

    def _init_state(self, img):
        self.rows, self.cols = img.shape[:2]
        self.x, self.y = np.random.randint(0, self.rows), np.random.randint(0, self.cols)
        self.angle = np.random.randint(0, 8) * 0.25 * np.pi
        self.max_len = self.rows // 8
        self.max_speed = self.max_len // 2
        self.speed = np.random.randint(1, self.max_speed)
        self.intensity = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.paths = []
        self.values = []
        self.prev_row_vals = []
        self.frame = 0
        self.initialized = True

    def step(self, img):
        # One iteration of the random mover
        sorted_img = img.copy()
        # Move in a direction with speed
        dx = int(round(np.cos(self.angle) * self.speed))
        dy = int(round(np.sin(self.angle) * self.speed))
        x1 = self.x + dx
        y1 = self.y + dy

        #print((x, y), (x1, y1), angle, speed)

        # Bounce off walls
        if x1 < 0 or x1 >= self.rows or y1 < 0 or y1 >= self.cols:
            # Reverse direction
            dx = -dx
            dy = -dy
            #x1 = x + dx
            #y1 = y + dy
            self.angle = np.arctan2(dy, dx)

        # Ensure end point is within bounds
        x1 = max(0, min(x1, self.rows - 1))
        y1 = max(0, min(y1, self.cols - 1))

        # Perpendicular direction (normalize to unit vector)
        perp_angle = self.angle + np.pi / 2

        # Get all pixels along the line
        line_pixels = list(bresenham_line(self.x, self.y, x1, y1))
        
        # prev_row_vals = []
        for px, py in line_pixels:        
            #print(current_val, prev_val, val_change, (val_change >= threshold), len(path))

            current_row_vals = []
            current_row = []
            for w in range(-(self.brush_width//2), self.brush_width//2 + 1):
                qx = int(round(px + w * np.cos(perp_angle)))
                qy = int(round(py + w * np.sin(perp_angle)))

                # Ensure points are within bounds
                qx = max(0, min(qx, self.rows - 1))
                qy = max(0, min(qy, self.cols - 1))

                current_row.append((qx, qy))
                current_row_vals.append(int(self.intensity[qx, qy]))

            self.paths.append(current_row)
            self.values.append(current_row_vals)

            #if len(self.prev_row_vals) != 0:
            #    val_change = np.mean(np.abs(np.array(current_row_vals) - np.array(self.prev_row_vals)))
            #else:
            #    val_change = 0
            #self.prev_row_vals = current_row_vals
            val_change = np.mean(np.array(self.values).max(axis=0) - np.array(self.values).min(axis=0))

            if (val_change >= self.threshold) or (len(self.paths[0]) >= self.max_len):
                if self.set_color:
                    self.color = colors[np.random.randint(0, len(colors))]
                
                # loop over paths 
                #paths = np.array(self.paths).T
                self.values = np.array(self.values)
                
                # sort value along the path
                if not self.set_color:
                    sorted_indices = np.argsort(self.values, axis=0).T
      
                for path_i in range(len(self.paths[0])):
                    # Get the path and values for the current path index
                    path = [row[path_i] for row in self.paths]
                    for step, (qx, qy) in enumerate(path):
                        if self.set_color:
                            sorted_img[qx, qy] = color
                        else:
                            qx1, qy1 = path[sorted_indices[path_i, step]]
                            sorted_img[qx, qy] = img[qx1, qy1]

                # use the sorted values if uncommented      
                #intensity = cv2.cvtColor(sorted_img, cv2.COLOR_BGR2GRAY)
                
                # Reset path and values
                self.paths = []
                self.values = []

        # Update current position to end of line
        self.x, self.y = x1, y1

        # change angle and speed
        angle_change = np.random.randint(-1, 2) * np.pi / 4
        self.angle += angle_change
        speed_change = np.random.randint(-1, 3)
        self.speed += speed_change

        # ensure angle is within bounds
        if self.angle < 0:
            self.angle += 2 * np.pi
        elif self.angle > 2 * np.pi:
            self.angle -= 2 * np.pi

        # ensure speed is within bounds
        if self.speed < 1:
            self.speed = 1
        elif self.speed > self.max_speed:
            self.speed = self.max_speed

        return sorted_img

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

class PixelSorter:
    def __init__(self, img, artists=None):
        self.sorted_img = img.copy()
        self.img = img
        self.artists = artists if artists is not None else []

    def add_artist(self, artist):
        self.artists.append(artist)

    def draw(self, iterations=1, show=True, skip=10):
        # Run one iteration of each artist
        for artist in self.artists:
            if not artist.initialized:
                artist._init_state(self.img)

            for i in tqdm(range(iterations)):
                self.sorted_img = artist.step(self.sorted_img)

                # Display the image every 10 iterations
                if show and i % skip == 0:
                    cv2.imshow('Sorted Image', self.sorted_img)
                    cv2.waitKey(1)

            self.img = self.sorted_img
        
        return self.sorted_img

# Utility function
def bresenham_line(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            yield x, y
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
        yield x, y
    else:
        err = dy / 2.0
        while y != y1:
            yield x, y
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
        yield x, y

def main(img_path='nj.png'):
    img = cv2.imread(img_path)
    scale = 0.5
    resized_img = cv2.resize(img, dsize=(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Example usage:
    artists = [
        MultiHeadedRandomMoverArtist(max_iter=99999, brush_width=40, threshold=120, num_heads=2),
        #GridArtist(rows=24, cols=9, direction='horizontal'),
        #GridArtist(rows=12, cols=8, direction='vertical'),
    ]
    sorter = PixelSorter(resized_img, artists=artists)

    # Run for a number of steps
    img = sorter.draw(iterations=1500, show=False, skip=50)

    cv2.imwrite('sorted.png', img)
    cv2.imshow('Sorted Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tyro.cli(main)