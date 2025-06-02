import cv2
import numpy as np
from tqdm import tqdm
import tyro
from artist import Artist, RandomLineArtist, GridArtist, MultiHeadedRandomMoverArtist, RandomMoverArtist

class PixelSorter:
    def __init__(self, img, artists=None):
        self.sorted_img = img.copy()
        self.img = img
        self.artists = artists if artists is not None else []

    def add_artist(self, artist):
        self.artists.append(artist)

    def draw(self, iterations=1, show=True, skip=1):
        # Run one iteration of each artist
        for artist in self.artists:
            if not artist.initialized:
                artist._init_state(self.img)

            for i in tqdm(range(iterations)):
                artist.step(self.sorted_img)

                # Display the image every 10 iterations
                if show and i % skip == 0:
                    cv2.imshow('Sorted Image', self.sorted_img)
                    cv2.waitKey(1)

            #self.img = self.sorted_img
        
        return self.sorted_img

def main(img_path='nj.png'):
    img = cv2.imread(img_path)
    scale = 0.5
    resized_img = cv2.resize(img, dsize=(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Example usage:
    artists = [
        #RandomMoverArtist(max_iter=1500, brush_width=40, threshold=150),
        #GridArtist(rows=24, cols=9, direction='horizontal'),
        #GridArtist(rows=12, cols=8, direction='vertical'),
        RandomLineArtist(brush_width=40, max_len=175, min_len=50, min_angle=np.deg2rad(0), max_angle=np.deg2rad(360), set_color=False)
    ]
    sorter = PixelSorter(resized_img, artists=artists)

    # Run for a number of steps
    img = sorter.draw(iterations=1500, show=True, skip=10)

    cv2.imwrite('sorted.png', img)
    cv2.imshow('Sorted Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tyro.cli(main)