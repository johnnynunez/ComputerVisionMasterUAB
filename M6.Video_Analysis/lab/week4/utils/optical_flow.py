# Optical flow estimations using the Lucas-Kanade algorithm.
# Sequences 45 and 157 (image_0) from the KITTI dataset.
# Only 1 estimation / sequence (2 frames!)
# Check the KITTI website for code to read results (dense motion vectors)
import os

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def flow_read(filename):
    """Read optical flow from png file (adapted version from KITTI development tools kit).
    Args:
        filename: path to optical flow png file.
    Returns:
        F: optical flow.
    """
    I = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.double)

    F_u = (I[:, :, 2] - 2 ** 15) / 64
    F_v = (I[:, :, 1] - 2 ** 15) / 64

    # check if there exists a valid GT flow for that pixel (1: True, 0: False)
    F_valid = I[:, :, 0]
    F_valid[F_valid > 1] = 1

    F_u[F_valid == 0] = 0
    F_v[F_valid == 0] = 0

    F = np.dstack((F_u, F_v, F_valid))
    return F


def compute_errors(pred_flow, gt_flow, threshold, save_path, plots=False):
    """Compute the mean squared error in Non-occluded (MSEN) areas and the percentage of erroneous pixels (PEPN).
       Visualize the results if plots = True.
    Args:
        pred_flow: predicted optical flow.
        gt_flow: ground truth optical flow.
        threshold: threshold for the error in the PEPN.
    Returns:
        msen: mean squared error.
        pepn: percentage of erroneous pixels.
    """

    diff_u = gt_flow[:, :, 0] - pred_flow[:, :, 0]
    diff_v = gt_flow[:, :, 1] - pred_flow[:, :, 1]

    sq_diff = np.sqrt(diff_u ** 2 + diff_v ** 2)
    sq_diff_valid = sq_diff[gt_flow[:, :, 2] == 1]

    msen = np.mean(sq_diff_valid)
    pepn = (np.sum(sq_diff_valid > threshold) / len(sq_diff_valid)) * 100

    # visualizations

    if plots:
        # plot the error flow
        plt.imshow(sq_diff)
        plt.title("Optical Flow error")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.show()

        # plot the error histogram

        cm = plt.cm.get_cmap('viridis')

        bins = np.arange(np.floor(sq_diff_valid.min()), np.ceil(sq_diff_valid.max()))
        Y, X = np.histogram(sq_diff_valid, bins=bins, density=1)
        x_span = X.max() - X.min()
        C = [cm((x - X.min()) / x_span) for x in X]
        plt.bar(X[:-1], Y, width=X[1] - X[0], color=C, edgecolor='white', linewidth=0.2)
        plt.title('Error probability distribution')
        plt.xlabel('Error')
        plt.ylabel('Pixels probablity')
        plt.axvline(msen, color='orange', linestyle='dashed', label="MSEN", linewidth=1)
        plt.legend(loc='upper right')
        plt.show()

        # create otput folder if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save the plot
        plt.savefig(save_path + 'error_histogram.png')

    return msen, pepn


def opticalFlow_arrows(frame, flow_gt, flow, save_path, name):
    """Compute the optical flow arrows diagram.
    Args:
        frame: image.
        flow: optical flow.
    """
    step = 15
    height_gt, width_gt, _ = flow_gt.shape
    X_gt, Y_gt = np.meshgrid(np.arange(0, width_gt), np.arange(0, height_gt))

    # X, Y define the arrow locations, U, V define the arrow directions, and C optionally sets the color.
    U_gt = flow_gt[:, :, 0]
    V_gt = flow_gt[:, :, 1]
    M_gt = np.hypot(U_gt, V_gt)  # magnitude
    X_gt = X_gt[::step, ::step]
    Y_gt = Y_gt[::step, ::step]
    U_gt = U_gt[::step, ::step]
    V_gt = V_gt[::step, ::step]
    M_gt = M_gt[::step, ::step]

    height, width, _ = flow.shape
    step = 20
    X, Y = np.meshgrid(np.arange(0, width), np.arange(0, height))

    # X, Y define the arrow locations, U, V define the arrow directions, and C optionally sets the color.
    U = flow[:, :, 0]
    V = flow[:, :, 1]
    M = np.hypot(U, V)  # magnitude
    X = X[::step, ::step]
    Y = Y[::step, ::step]
    U = U[::step, ::step]
    V = V[::step, ::step]
    M = M[::step, ::step]

    max_value = max(M_gt.max(), M.max())
    norm = mcolors.Normalize(vmin=0, vmax=max_value)

    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(20, 10)
    im0 = ax[0].imshow(frame, cmap="gray")
    im0 = ax[0].quiver(
        X_gt,
        Y_gt,
        U_gt,
        V_gt,
        M_gt,
        scale_units="xy",
        angles="xy",
        pivot="mid",
        scale=0.1,
        norm=norm,
        cmap="rainbow",
    )
    im1 = ax[1].imshow(frame, cmap="gray")
    im1 = ax[1].quiver(
        X,
        Y,
        U,
        V,
        M,
        scale_units="xy",
        angles="xy",
        pivot="mid",
        scale=0.1,
        norm=norm,
        cmap="rainbow",
    )

    ax[0].set_title("GT")
    ax[1].set_title("Predicted")
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    fig.suptitle("GT vs Predicted Optical Flow")
    fig.colorbar(im0, ax=ax.ravel().tolist())
    # plt.show()

    # create otput folder if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save the plot
    plt.savefig(save_path + 'OF_arrows_' + name + '.png')


"""def HSVOpticalFlow(flow, title, save_path):
    "version of opticalflow cv2 optical flow example"
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # clip the magnitude to 0.95 quantile to remove outliers / better visualization not affected by extreme values when noemalizing
    clip = np.quantile(magnitude, 0.95)
    mag = np.clip(mag, 0, clip)

    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # magnitude and angle
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # only the angle
    hsv2 = hsv
    hsv2[..., 2] = 255
    rgb2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)

    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(50, 10)
    ax[2].imshow(rgb)
    ax[0].imshow(mag, cmap="gray")
    ax[1].imshow(rgb2)
    ax[2].set_title("Magnitude and Angle")
    ax[0].set_title("Magnitude")
    ax[1].set_title("Angle")
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    fig.suptitle(title)
    plt.show()

    #create otput folder if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    #save the plot
    plt.savefig(save_path + 'optical_flow_hsv.png')"""


def HSVOpticalFlow2(flow, save_path, name):
    "version of opticalflow cv2 optical flow example changing S and V for better visualization"
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # clip the magnitude to 0.95 quantile to remove outliers / better visualization not affected by extreme values when noemalizing
    clip = np.quantile(magnitude, 0.95)
    mag = np.clip(mag, 0, clip)

    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = 255
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    fig = plt.figure(figsize=(20, 10))
    plt.imshow(rgb)
    plt.title("Optical Flow HSV")
    plt.axis("off")
    # change size

    # plt.show()

    # create otput folder if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save the plot
    plt.savefig(save_path + 'OF_hsv_' + name + '.png')

    # close the plot
    plt.close(fig)
