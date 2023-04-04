import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
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

img_size = (256, int(256 * 0.75))

image = cv2.imread("/content/000003_0.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(img_size),transforms.ToTensor()])

label = cv2.imread("/content/Self-Correction-Human-Parsing/outputs/000003_0.png")
label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
label = cv2.resize(label, img_size[::-1], interpolation=cv2.INTER_NEAREST)

label_transf = np.zeros((*img_size, len(semantic_labels)))
for i, color in enumerate(semantic_labels):
    label_transf[np.all(label == color, axis=-1), i] = 1.0

label_transf = torch.tensor(label_transf, dtype=torch.float32).permute(2, 0, 1).contiguous()

# Create the mask for the head (Hair and Face)
parse_head = label_transf[2, :, :] + label_transf[3, :, :]
parse_head = parse_head.unsqueeze(0)

image = transform(image)
image = (image - 0.5) / 0.5

# Mask the image to get the desired head image
head_image = image * parse_head

# Convert head_image back to PIL Image, denormalize and save
head_image_pil = transforms.ToPILImage()(head_image * 0.5 + 0.5)
head_image_pil.save("/content/head_image.png")
