import math

import cv2
import numpy as np
import pandas as pd
from numpy import mgrid
from scipy.fftpack import fft2
from scipy.signal import convolve2d
from skimage.io import imsave
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import estimate_sigma


# STRATEGY 1
def estimate_noise_sigma(img):
    H, W = img.shape

    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(img, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W - 2) * (H - 2))

    return sigma


def sigma_grid(img):
    partitions = 16
    grid_image_sigma = 0
    height = img.shape[0]
    width = img.shape[1]

    newH = height // partitions
    newW = width // partitions

    for y in range(0, height, newH):
        for x in range(0, width, newW):
            partialImg = img[y:y + newH, x:x + newW]
            grid_image_sigma += estimate_noise_sigma(partialImg)

    return grid_image_sigma / (partitions * partitions)


# STRATEGY 2
def get_grids(N_X, N_Y):
    return mgrid[-1:1:1j * N_X, -1:1:1j * N_Y]


def frequency_radius(fx, fy):
    R2 = fx ** 2 + fy ** 2
    (N_X, N_Y) = fx.shape
    R2[int(N_X / 2), int(N_Y / 2)] = np.inf

    return np.sqrt(R2)


def enveloppe_color(fx, fy, alpha=1.0):
    # 0.0, 0.5, 1.0, 2.0 are resp. white, pink, red, brown noise
    # (see http://en.wikipedia.org/wiki/1/f_noise )
    # enveloppe
    return 1. / frequency_radius(fx, fy) ** alpha  #


def fft_denoising(img):
    power_spectrum = np.abs(fft2(img)) ** 2
    return power_spectrum


# STRATEGY 3

def estimate_noise(img):
    return estimate_sigma(img, channel_axis=-1, average_sigmas=True)


# type of noise
def noise_type(img):
    noise = estimate_noise(img)
    if noise < 0.05:
        return "gaussian"
    elif noise < 0.1:
        return "salt & pepper"
    elif noise < 0.2:
        return "speckle"
    else:
        return "unknown"


def average_filter(img):
    kernel = np.ones((3, 3), np.float32) / (3 * 3)
    return cv2.filter2D(img, -1, kernel)


def gaussian_filter(img):
    return cv2.GaussianBlur(img, (3, 3), 0.5)


def median_filter(img):
    return cv2.medianBlur(img, 3)


def bilateral_filter(img):
    sigmaColor = 100
    sigmaSpace = 5000
    d = 25
    borderType = cv2.BORDER_REPLICATE

    denoised_img = cv2.bilateralFilter(
        img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace, borderType=borderType, dst=None)
    return denoised_img


def laplacian_filter(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def laplacion_of_gaussian(img):
    kernel = cv2.getGaussianKernel(3, 0)
    kernel = kernel * kernel.T
    return cv2.filter2D(img, -1, kernel)


def normalize_img(img):
    normalized_img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return normalized_img


def compare(inp_img, out_img, calc_ssim=True, calc_msssim=True, calc_psnr=True, calc_mse=True):
    assert inp_img.shape == out_img.shape

    def get_ssim():
        return ssim(inp_img, out_img, channel_axis=-1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                    data_range=255)

    def get_msssim():
        return ssim(inp_img, out_img, data_range=255, channel_axis=-1, gaussian_weights=True,
                    use_sample_covariance=False)

    def get_psnr():
        return peak_signal_noise_ratio(inp_img, out_img, data_range=255)

    def get_mse():
        return mean_squared_error(inp_img, out_img)

    def _run_if(cond, fn):
        return fn() if cond else None

    return _run_if(calc_ssim, get_ssim), _run_if(calc_msssim, get_msssim), _run_if(calc_psnr, get_psnr), _run_if(
        calc_mse, get_mse)


def evaluate_metrics(img_orig, img_denoised, calc_ssim=True, calc_msssim=True, calc_psnr=True, calc_mse=True):
    ssim_list = []
    msssim_list = []
    psnr_list = []
    mse_list = []
    ssim, msssim, psnr, mse = compare(img_orig, img_denoised, calc_ssim, calc_msssim, calc_psnr, calc_mse)
    ssim_list.append(ssim)
    msssim_list.append(msssim)
    psnr_list.append(psnr)
    mse_list.append(mse)

    return ssim_list, msssim_list, psnr_list, mse_list


def evaluate_best_denoise(img_list, sigma_threshold):
    final_images = list()
    # create dataframe
    df = pd.DataFrame(columns=['image', 'sigma', 'sigma_denoised', 'ssim', 'msssim', 'psnr', 'mse', 'best_denoise'])
    for idx, img in enumerate(img_list):
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        original_sigma = estimate_sigma(img, average_sigmas=True, channel_axis=-1)
        print("Image Original {} sigma: {}".format(idx, original_sigma))
        if original_sigma >= sigma_threshold:
            denoised_img_list = list()
            sigmas_list = list()
            sigmas_denoised_list = list()
            ssim_list = list()
            psnr_list = list()
            msssim_list = list()
            mse_list = list()
            for denoiser in DENOISER_METHODS:
                denoised_img = DENOISER_METHODS[denoiser](img)
                # denoised_img_list.append(cv2.cvtColor(denoised_img, cv2.COLOR_GRAY2RGB))
                denoised_img_list.append(denoised_img)
                sigma_denoised = estimate_sigma(denoised_img, average_sigmas=True, channel_axis=-1)
                sigmas_denoised_list.append(sigma_denoised)
                img_2 = img.copy() - denoised_img
                ssim, msssim, psnr, mse = compare(img, img_2)
                print(f"{denoiser} SSIM: {ssim} MSSSIM: {msssim} PSNR: {psnr} MSE: {mse}")
                ssim_list.append(ssim)
                psnr_list.append(psnr)
                msssim_list.append(msssim)
                mse_list.append(mse)
                current_sigma = estimate_sigma(img, average_sigmas=True, channel_axis=-1)
                sigma_delta = original_sigma - current_sigma
                sigmas_list.append(sigma_delta)
                print(f"{denoiser} sigma: {current_sigma} sigma delta: {sigma_delta}")
            best_denoiser_ssim = list(DENOISER_METHODS.keys())[ssim_list.index(max(ssim_list))]
            best_denoiser_psnr = list(DENOISER_METHODS.keys())[psnr_list.index(max(psnr_list))]
            best_sigma_delta = list(DENOISER_METHODS.keys())[sigmas_list.index(max(sigmas_list))]
            print(f"Best denoiser for image {idx} is {best_denoiser_ssim}")
            print(f"Best denoiser for image {idx} is {best_denoiser_psnr}")
            print(f"Best sigma delta for image {idx} is {best_sigma_delta}")
            final_images.append(denoised_img_list[psnr_list.index(max(psnr_list))])
            df = df.append({'image': idx, 'sigma': original_sigma, 'sigma_denoised': sigmas_denoised_list[1],
                            'ssim': max(ssim_list), 'msssim': max(msssim_list), 'psnr': max(psnr_list),
                            'mse': min(mse_list), 'best_denoise': best_denoiser_psnr}, ignore_index=True)
        else:
            final_images.append(img)
            df = df.append({'image': idx, 'sigma': original_sigma, 'sigma_denoised': 0,
                            'ssim': None, 'msssim': None, 'psnr': None,
                            'mse': None, 'best_denoise': None}, ignore_index=True)
    df.to_csv('metrics.csv')
    return final_images


DENOISER_METHODS = {
    "gaussian_denoiser": gaussian_filter,
    "median_denoiser": median_filter,
    "average_denoiser": average_filter,
    "bilateral_denoiser": bilateral_filter,

}

if __name__ == "__main__":
    print("Denoising")
    from data_utils import DataHandler

    data_handler = DataHandler(n_process=8)
    qsd1_w4, qsd1_w4_files = data_handler.load_images(folder="./data/Test/qst1_w5/", extension=".jpg",
                                                      desc="Loading qsd1_w5 Data...")

    qsd1_w4_denoised = evaluate_best_denoise(qsd1_w4, 2)
    # save images
    for idx, img in enumerate(qsd1_w4_denoised):
        name = qsd1_w4_files[idx].split("/")[-1].split(".")[0]
        imsave(f"./data/Test/qst1_w5_denoised/{name}.jpg", img)
