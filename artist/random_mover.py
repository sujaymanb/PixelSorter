from artist import Artist
import numpy as np
import cv2

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