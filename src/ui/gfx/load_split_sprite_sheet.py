import cv2
import numpy as np
import pygame


class SplitSpriteSheet:

    def __init__(self, filename, frame_width, frame_height):

        self.screen = None
        self.filename = filename
        self.frame_width = frame_width
        self.frame_height = frame_height


        pygame.init()  # Alustetaan Pygame
        pygame.display.set_mode((1, 1))  # Tarvitaan, vaikka ikkunan kokoa ei käytetä
        sprite_sheet = pygame.image.load(filename).convert_alpha()

        self.frames = []
        sheet_width, sheet_height = sprite_sheet.get_size()
        cols = sheet_width // frame_width
        rows = sheet_height // frame_height

        for row in range(rows):
            for col in range(cols):
                frame = sprite_sheet.subsurface((col * frame_width, row * frame_height, frame_width, frame_height))
                self.frames.append(frame)

    def draw_sprite(self, x, y):


        frame_rgb = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2RGB)

        # 2. Rotate frame if necessary (OpenCV and Pygame handle axes differently)
        frame_rgb = np.rot90(frame_rgb)  # Optional: Use if the frame is rotated incorrectly

        # 3. Convert NumPy array to a Pygame surface
        frame_surface = pygame.surfarray.make_surface(frame_rgb)

        # 4. Blit the surface onto the Pygame screen
        self.screen.blit(frame_surface, (0, 0))  # Adjust position as needed

        # 5. Update the Pygame display
        pygame.display.flip()




