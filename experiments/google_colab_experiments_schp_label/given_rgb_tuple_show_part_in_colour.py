import numpy as np
import cv2

def get_body_part_image(label, image, rgb_tuple):
    # Create a mask for the given RGB tuple
    body_part_mask = np.all(label == rgb_tuple, axis=-1)

    # Create a 3-channel mask
    body_part_mask_3channel = np.stack([body_part_mask, body_part_mask, body_part_mask], axis=-1)

    # Apply the mask to the original image
    body_part_image = np.where(body_part_mask_3channel, image, 0)

    return body_part_image

# Read the label image and resize it to the specified size
label = cv2.imread("/content/Self-Correction-Human-Parsing/outputs/000003_0.png")
label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
label = cv2.resize(label, img_size[::-1], interpolation=cv2.INTER_NEAREST)

# Read the original image and resize it to the specified size
image = cv2.imread("/content/000003_0.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, img_size[::-1], interpolation=cv2.INTER_AREA)

# Specify the RGB tuple for the body part you want to display
rgb_tuple = (0,128,0)  # Example: Hair

# Get the body part image
body_part_image = get_body_part_image(label, image, rgb_tuple)

# Save the body part image
cv2.imwrite("/content/body_part_image.png", cv2.cvtColor(body_part_image, cv2.COLOR_RGB2BGR))