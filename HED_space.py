import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#RanS 2.12.20
RGB_SCALE = 255
CMYK_SCALE = 255


def rgb_to_cmyk(r, g, b):
    if (r, g, b) == (0, 0, 0):
        # black
        return 0, 0, 0, CMYK_SCALE

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / RGB_SCALE
    m = 1 - g / RGB_SCALE
    y = 1 - b / RGB_SCALE

    # extract out k [0, 1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    # rescale to the range [0,CMYK_SCALE]
    return c * CMYK_SCALE, m * CMYK_SCALE, y * CMYK_SCALE, k * CMYK_SCALE


def cmyk_to_rgb(c, m, y, k, cmyk_scale, rgb_scale=255):
    r = rgb_scale * (1.0 - c / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    g = rgb_scale * (1.0 - m / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    b = rgb_scale * (1.0 - y / float(cmyk_scale)) * (1.0 - k / float(cmyk_scale))
    return r, g, b


# RanS 25.11.20
def RGB2HED(image, eps=1e-10):
    # Convert into H&E color space
    M1 = np.array([[0.18, 0.20, 0.08],
                   [0.01, 0.13, 0.01],
                   [0.10, 0.21, 0.29]])
    #M = M1/np.linalg.norm(M1,axis=1).reshape([3,1])

    D = np.array([[ 1.88, -0.07, -0.60],
                  [-1.02,  1.13, -0.48],
                  [-0.55, -0.13,  1.57]])

    M = np.array([[0.65, 0.70, 0.29],
                  [0.07, 0.99, 0.11],
                  [0.27, 0.57, 0.78]])

    M_inverse = np.array([[1.87798274, -1.00767869, -0.55611582],
                        [-0.06590806, 1.13473037, -0.1355218],
                        [-0.60190736, -0.48041419, 1.57358807]])

    #plot_image_in_channels(image, 'original image')
    image = 255 - image #RanS 26.11.20, absorption is 1-transfer
    #plot_image_in_channels(image, '255-image')

    image = image/255
    #plot_image_in_channels(image, 'normalized to 1')

    #image_vec = image.reshape(image.shape[0]*image.shape[1], 3)
    #eps = 1e-10#5

    '''for ii in range(3):
            print('min image[:,:,', str(ii), ']:', np.min(image[:, :, ii]))
            print('max image[:,:,', str(ii), ']:', np.max(image[:, :, ii]))'''

    OD_image = -np.log(image + eps)
    #plot_image_in_channels(OD_image, 'OD_image')
    OD_image_vec = image.reshape(OD_image.shape[0] * OD_image.shape[1], 3)
    # image_vec_HED_flat = (-np.log(image_vec+eps)) @ np.linalg.inv(M)
    # image_vec_HED_flat = (-np.log(image_vec + eps)) @ M_inverse
    #image_vec_HED_flat = (-np.log(image_vec + eps)) @ D
    OD_image_vec_HED = OD_image_vec @ D
    # image_vec_HED_flat = (-np.log(image_vec+eps)) @ D.T
    OD_image_HED = OD_image_vec_HED.reshape(image.shape[0], image.shape[1], 3)
    #plot_image_in_channels(OD_image, 'OD_image_HED')

    # temp RanS 26.11.20
    #image_vec_HED = np.exp(-OD_image_vec_HED) - eps
    image_vec_HED = np.exp(-OD_image_HED) - eps

    image_HED = image_vec_HED.reshape(image.shape[0], image.shape[1], 3)

    #plot_image_in_channels(image_HED, 'image_HED')

    image_HED = 1 - image_HED

    #plot_image_in_channels(image_HED, 'final, 1 - image_HED')
    #plt.show()
    return image_HED

# RanS 25.11.20
def HED2RGB(image_HED, eps = 1e-10):
    # Convert into H&E color space

    D_inv = np.array([[0.64961187, 0.07131043, 0.27006123],
                  [0.70794297, 0.99493047, 0.57473402],
                  [0.28619052, 0.10736414, 0.77913955]])

    #plot_image_in_channels(image_HED, 'image_HED')
    image_HED = 1 - image_HED #temp cancelled RanS 29.11.20
    #plot_image_in_channels(image_HED, '1-image_HED')

    #eps = 1e-10  # 5
    OD_image_HED = -np.log(image_HED + eps)
    #plot_image_in_channels(OD_image_HED, 'OD_image_HED')
    OD_image_vec_HED = OD_image_HED.reshape(OD_image_HED.shape[0] * OD_image_HED.shape[1], 3)
    OD_image_vec = OD_image_vec_HED @ D_inv
    OD_image = OD_image_vec.reshape(OD_image_HED.shape[0], OD_image_HED.shape[1], 3)
    #plot_image_in_channels(OD_image, 'OD_image')
    image = np.exp(-OD_image) - eps
    #clip where saturated
    image[image > 1] = 1
    image[image < 0] = 0
    #plot_image_in_channels(image, 'image')
    image = (image*255).astype(np.uint8)
    #plot_image_in_channels(image, 'image*255')
    #image = 255 - image
    #plot_image_in_channels(image, '255-image')
    return image


def HED_color_jitter(image, sigma=0.05):
    HED_image = RGB2HED(image)
    HED_image_tag = np.zeros_like(HED_image)
    for ii in range(3):
        alpha_channel = 1 - sigma + 2 * sigma * np.random.rand()
        beta_channel = -sigma + 2 * sigma * np.random.rand()
        HED_image_tag[:, :, ii] = alpha_channel * HED_image[:, :, ii] + beta_channel
    image_tag = HED2RGB(HED_image_tag)
    return image_tag


def plot_image_in_channels(img, title):
    fig, axs = plt.subplots(1, 4)
    if np.min(img)<0: #need to rescale color image
        img_scaled = (img - np.min(img)) / (np.max(img) - np.min(img))
    else:
        img_scaled = img
    axs[0].imshow(img_scaled)
    axs[0].set_title('original')

    color_min = np.min(img)
    color_max = np.max(img)
    for ii,titl in zip(range(3), ['r', 'g', 'b']):
        im1 = axs[ii+1].imshow(np.squeeze(img[:, :, ii]), vmin=color_min, vmax=color_max)
        divider1 = make_axes_locatable(axs[ii+1])
        cax1 = divider1.append_axes("right", size="20%", pad=0.05)
        plt.colorbar(im1, cax=cax1)
        axs[ii+1].set_title(titl)
    fig.suptitle(title)