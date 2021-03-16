# %% --------------------
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

from common.utilities import get_image_as_array

# %% --------------------

img = get_image_as_array(
    'D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm '
    'Workspace/vbd_cxr/9_data/512/transformed_data/train/0c4a6bc602d1d207f217212c68a7131b.jpeg')
img = np.asarray(img)
plt.figure(figsize=(12, 12))
plt.imshow(img, 'gray')
plt.show()

# %% --------------------
img_hist = exposure.equalize_hist(img)
plt.figure(figsize=(12, 12))
plt.imshow(img_hist, 'gray')
plt.show()
# %% --------------------
img_clahe = exposure.equalize_adapthist(img / np.max(img))
plt.figure(figsize=(12, 12))
plt.imshow(img_clahe, 'gray')
plt.show()