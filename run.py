import pydicom
import numpy as np
from matplotlib import pyplot as plt


def load_dcm(filename):
    return pydicom.dcmread(f'data/{filename}')


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


if __name__ == '__main__':
    main()
