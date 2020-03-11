import pydicom
import numpy as np
from matplotlib import pyplot as plt, animation
from scipy import ndimage


def load_dcm(filename):
    return pydicom.dcmread(f'data/{filename}')


def rotate_YZ_v1(image, angle_in_degrees):
    return ndimage.rotate(image, angle_in_degrees, axes=(1, 2), reshape=False)


def rotate_YZ_v2(image, angle_in_degrees, background_padding_value):
    y_grid, z_grid = np.meshgrid(np.linspace(0, image.shape[1], num=image.shape[1]),
                                 np.linspace(0, image.shape[2], num=image.shape[2]))

    rotated_y_grid = np.sin(angle_in_degrees) * (y_grid - image.shape[1] / 2) + np.cos(angle_in_degrees) * (z_grid - image.shape[2] / 2)
    rotated_y_grid = np.round(rotated_y_grid + image.shape[1] / 2)
    out_of_bounds_y = np.logical_or(rotated_y_grid < 0, rotated_y_grid >= image.shape[1])
    rotated_y_grid = np.minimum(np.maximum(rotated_y_grid, 0), image.shape[1] - 1)

    rotated_z_grid = np.cos(angle_in_degrees) * (y_grid - image.shape[1] / 2) - np.sin(angle_in_degrees) * (z_grid - image.shape[2] / 2)
    rotated_z_grid = np.round(rotated_z_grid + image.shape[2] / 2)
    out_of_bounds_z = np.logical_or(rotated_y_grid < 0, rotated_y_grid >= image.shape[2])
    rotated_z_grid = np.minimum(np.maximum(rotated_z_grid, 0), image.shape[2] - 1)

    rotated_image = image[:, rotated_y_grid.astype('int'), rotated_z_grid.astype('int')]
    rotated_image[:, out_of_bounds_y] = background_padding_value
    rotated_image[:, out_of_bounds_z] = background_padding_value

    return rotated_image


def main():
    dcm = load_dcm(filename='16351644_s1_CT_PETCT.dcm')
    print(dcm)

    img = np.flip(dcm.pixel_array, axis=0)
    pixel_len_mm = [3.27, 0.98, 0.98]

    plano_medio_coronal = img[:, img.shape[1]//2, :]
    plano_medio_sagital = img[:, :, img.shape[2]//2]

    plt.subplot(1, 2, 1)
    plt.imshow(plano_medio_sagital, cmap=plt.cm.get_cmap('bone'), aspect=pixel_len_mm[0]/pixel_len_mm[1])
    plt.subplot(1, 2, 2)
    plt.imshow(plano_medio_coronal, cmap=plt.cm.get_cmap('bone'), aspect=pixel_len_mm[0]/pixel_len_mm[2])
    plt.show()

    # Proyecciones MIP / AIP coronales
    coronal_MIP = np.max(img, axis=1)
    coronal_AIP = np.mean(img, axis=1)
    plt.subplot(1, 2, 1)
    plt.imshow(coronal_MIP, cmap=plt.cm.get_cmap('bone'), aspect=pixel_len_mm[0]/pixel_len_mm[2])
    plt.subplot(1, 2, 2)
    plt.imshow(coronal_AIP, cmap=plt.cm.get_cmap('bone'), aspect=pixel_len_mm[0]/pixel_len_mm[2])
    plt.show()

    # Proyecciones MIP / AIP coronales
    sagital_MIP = np.max(img, axis=2)
    sagital_AIP = np.mean(img, axis=2)
    plt.subplot(1, 2, 1)
    plt.imshow(sagital_MIP, cmap=plt.cm.get_cmap('bone'), aspect=pixel_len_mm[0]/pixel_len_mm[1])
    plt.subplot(1, 2, 2)
    plt.imshow(sagital_AIP, cmap=plt.cm.get_cmap('bone'), aspect=pixel_len_mm[0]/pixel_len_mm[1])
    plt.show()

    # Proyecciones MIP tras rotación en el plano axial (YZ).
    #       Colormap config
    img_min = np.amin(img)
    img_max = np.amax(img)
    cm = plt.cm.get_cmap('bone')
    cm_min = img_min
    cm_max = img_max

    n = 5
    for idx, alpha in enumerate(np.linspace(0, 2*np.pi*(n-1)/n, num=n)):
        rotated_img = rotate_YZ_v2(img, alpha, background_padding_value=-1000)
        sagital_MIP = np.amax(rotated_img, axis=2)
        plt.subplot(1, n, idx+1), plt.imshow(sagital_MIP, cmap=cm, vmin=cm_min, vmax=cm_max,
                                             aspect=pixel_len_mm[0]/pixel_len_mm[1])
    plt.show()


if __name__ == '__main__':
    main()
