import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# Function: Intensity Shift (brightness adjustment)
def intensity_shift(image, shift_factor=0.2):
    # shift_factor: fraction of max-min range to shift intensity by (+ or -)
    min_val, max_val = image.min(), image.max()
    shift = (max_val - min_val) * shift_factor
    img_shifted = np.clip(image + shift, min_val, max_val)
    return img_shifted

# Function: Elastic Deformation
def elastic_deformation(image, alpha=15, sigma=3):
    random_state = np.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).reshape(-1), (x + dx).reshape(-1)

    distorted_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    return distorted_image

# Function: Additive Speckle Noise
def speckle_noise(image, noise_level=0.1):
    # noise = noise_level * np.random.randn(*image.shape)
    # noisy_image = image + image * noise
    # return np.clip(noisy_image, image.min(), image.max())
    return gaussian_filter(image, sigma=2)

# -----------------------------
# 1. Read the image volume and mask
img = sitk.ReadImage("image_mha/03744.mha")
mask = sitk.ReadImage("label_mha/03744.mha")

arr = sitk.GetArrayFromImage(img)
mask_arr = sitk.GetArrayFromImage(mask)

if arr.ndim == 3:
    img_slice = arr[arr.shape[0] // 2]
else:
    img_slice = arr

if mask_arr.ndim == 3:
    mask_slice = mask_arr[mask_arr.shape[0] // 2]
else:
    mask_slice = mask_arr

# Normalize the image slice for augmentation and display (0-1 range)
img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())

# Apply augmentations
img_intensity = intensity_shift(img_norm, shift_factor=0.15)
img_elastic = elastic_deformation(img_norm, alpha=15, sigma=3)
img_speckle = speckle_noise(img_norm, noise_level=0.1)


# Prepare RGB mask visualization (same as before)
h, w = mask_slice.shape
rgb = np.zeros((h, w, 3), dtype=np.uint8)
rgb[mask_slice == 1] = [255, 0, 0]
rgb[mask_slice == 2] = [0, 255, 0]

# Function to create overlay image for display
def overlay_image(image_gray, mask_rgb, alpha=0.5):
    image_rgb = np.stack([image_gray]*3, axis=-1)
    overlay = image_rgb.copy()
    overlay[mask_rgb > 0] = (1 - alpha) * image_rgb[mask_rgb > 0] + alpha * (mask_rgb[mask_rgb > 0] / 255.0)
    return overlay

# Create overlay images for original and augmented
overlay_original = overlay_image(img_norm, rgb)
# overlay_intensity = overlay_image(img_intensity, rgb)
# overlay_elastic = overlay_image(img_elastic, rgb)
# overlay_speckle = overlay_image(img_speckle, rgb)

# Plot images
fig, axs = plt.subplots(1, 1, figsize=(4, 4))

# Original images
# axs[0, 0].imshow(img_norm, cmap='gray')
# axs[0, 0].set_title('Original Ultrasound')
# axs[0, 0].axis('off')

# axs[0, 1].imshow(rgb)
# axs[0, 1].set_title('Mask (PS=red, FH=green)')
# axs[0, 1].axis('off')

# axs[0, 2].imshow(overlay_original)
# axs[0, 2].set_title('Original + Mask Overlay')
# axs[0, 2].axis('off')


# axs.imshow(img_norm, cmap='gray')
# axs.axis('off')

# axs.imshow(rgb)
# axs.axis('off')

# axs.imshow(overlay_original)
# # axs[2].set_title('Original + Mask Overlay')
# axs.axis('off')
# Augmented images

# axs.imshow(img_speckle, cmap='gray')

# axs.axis('off')

# axs[1, 0].imshow(img_intensity, cmap='gray')
# axs[1, 0].set_title('Intensity Shift')
# axs[1, 0].axis('off')

# axs[1, 1].imshow(img_elastic, cmap='gray')
# axs[1, 1].set_title('Elastic Deformation')
# axs[1, 1].axis('off')

# axs[1, 2].imshow(img_speckle, cmap='gray')
# axs[1, 2].set_title('Speckle Noise')
# axs[1, 2].axis('off')

# plt.tight_layout()
# plt.show()

plt.imshow(img_speckle, cmap='gray')
plt.axis('off')
plt.show()


