from artist import Artist
import cv2
import numpy as np
from util.line_utils import bresenham_line

class RandomLineArtist(Artist):
    def __init__(self, 
        max_iter=2500, 
        brush_width=5, 
        max_len=500, 
        min_len=100, 
        min_angle=0, 
        max_angle=2*np.pi, 
        set_color=False
    ):
        self.max_iter = max_iter
        self.brush_width = brush_width
        self.initialized = False
        self.set_color = set_color
        self.max_len = max_len
        self.min_len = min_len
        self.min_angle = min_angle
        self.max_angle = max_angle

    def _init_state(self, img):
        self.rows, self.cols = img.shape[:2]
        self.update_line()
        self.colors = np.random.randint(0, 256, size=(100, 3), dtype=np.uint8)
        self.initialized = True
        #self.intensity = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #self.original_img = img.copy()

    def update_line(self):
        self.x = np.random.randint(0, self.rows)
        self.y = np.random.randint(0, self.cols)
        self.angle = np.random.uniform(self.min_angle, self.max_angle)
        self.length = np.random.randint(self.min_len, self.max_len)
        
    def get_line_vals(self, line_pixels, img):
        paths = []
        values = []
        perp_angle = self.angle + np.pi / 2
        intensity = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #intensity = self.intensity

        for px, py in line_pixels:
            current_row_vals = []
            current_row = []
            for w in range(-(self.brush_width//2), self.brush_width//2 + 1):
                qx = int(round(px + w * np.cos(perp_angle)))
                qy = int(round(py + w * np.sin(perp_angle)))

                # Ensure points are within bounds
                qx = max(0, min(qx, self.rows - 1))
                qy = max(0, min(qy, self.cols - 1))

                current_row.append((qx, qy))
                current_row_vals.append(int(intensity[qx, qy]))

            paths.append(current_row)
            values.append(current_row_vals)

        return paths, values

    def step(self, img):
        if not self.initialized:
            self._init_state(img)

        # Calculate end point of the line
        dx = int(round(np.cos(self.angle) * self.length))
        dy = int(round(np.sin(self.angle) * self.length))
        x1 = self.x + dx
        y1 = self.y + dy

        # Ensure end point is within bounds
        x1 = max(0, min(x1, img.shape[0] - 1))
        y1 = max(0, min(y1, img.shape[1] - 1))

        # Sort pixels along the line with thickness
        line_pixels = list(bresenham_line(self.x, self.y, x1, y1))
        paths, values = self.get_line_vals(line_pixels, img)
        values = np.array(values)
        
        if not self.set_color:
            sorted_indices = np.argsort(values, axis=0).T
        else:
            color = self.colors[np.random.randint(0, len(self.colors))]
        
        img_presort = img.copy()
        for path_i in range(len(paths[0])):
            # Get the path and values for the current path index
            path = [row[path_i] for row in paths]
            for step, (qx, qy) in enumerate(path):
                if self.set_color:
                    img[qx, qy] = color
                else:
                    qx1, qy1 = path[sorted_indices[path_i, step]]
                    img[qx, qy] = img_presort[qx1, qy1]

        self.update_line()