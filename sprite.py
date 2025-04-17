import cv2

def split_sprite_sheet(sprite_width, sprite_height, columns_in_sheet, rows_in_sheet, sprite_path):
    """Cuts the sprite sheet into individual frames."""
    sprite_sheet = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)

    sprite_frames = []
    for row in range(rows_in_sheet):
        for col in range(columns_in_sheet):
            x = col * sprite_width
            y = row * sprite_height
            sprite_frame = sprite_sheet[y:y + sprite_height, x:x + sprite_width]
            sprite_frames.append(sprite_frame)
    return sprite_frames


# Määritä frame-koko ja sarakkeiden sekä rivien määrä
frame_width = 196   # Vaihda oikeaan sprite-kokoon
frame_height = 190  # Vaihda oikeaan sprite-kokoon
cols = 13  # Kuinka monta saraketta sprite sheetissä on
rows = 1  # Kuinka monta riviä sprite sheetissä on

# Pilko sprite sheet animaation freimeiksi
frames = split_sprite_sheet(frame_width, frame_height, cols, rows, "./assets/explosion_strip13.png")

# Näytä animaatio OpenCV-ikkunassa
for frame in frames:
    cv2.imshow("Explosion Animation", frame)
    if cv2.waitKey(200) & 0xFF == ord('q'):  # Odota 100ms per frame
        break

cv2.destroyAllWindows()
