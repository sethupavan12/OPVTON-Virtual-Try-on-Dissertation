import numpy as np
import cv2

semantic_labels = [(0, 0, 0),    # 0=Background and SKIN
    (128, 0, 128),  # 1=Dress
    (0,128,0),  # 2=Hair
    (192,0,128),   # 3=Face
    (192,128,128),  # 4=left arm
    (64,128,128),  # 5=right arm
    (128,192,0),     # 6=left shoe
    (128,64,0),  # 7=left ankle
    (0,192,0),    # 8=right shoe
    (0,64,0),    # 9=right ankle
    (192,0,0),    # 10=lower body
]

# (128, 0, 128) - Dress
# (0,128,0) - hair
# (192,0,128) - face
# (192,128,128) - left arm
# (192,0,0) - lower body
# (64,128,128) - right arm
# (128,192,0) - left shoe
# (128,64,0) - left ankle
# (0,192,0) - right shoe
# (0,64,0 - right ankle



img_size = (256, int(256 * 0.75))

label = cv2.imread("/content/Self-Correction-Human-Parsing/outputs/000003_0.png")
label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
label = cv2.resize(label, img_size[::-1], interpolation=cv2.INTER_NEAREST)

# Read the original image and resize it to the specified size
image = cv2.imread("/content/000003_0.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, img_size[::-1], interpolation=cv2.INTER_AREA)

face_mask = (label == semantic_labels[2])
hair_mask = (label == semantic_labels[3])

# Combine the face and hair masks
head_mask = np.logical_or(face_mask[:, :, 0], hair_mask[:, :, 0])

# Create a 3-channel mask
head_mask_3channel = np.stack([head_mask, head_mask, head_mask], axis=-1)

# Apply the mask to the original image
head_image = np.where(head_mask_3channel, image, 0)

# Save the head image
cv2.imwrite("/content/head_image.png", cv2.cvtColor(head_image, cv2.COLOR_RGB2BGR))