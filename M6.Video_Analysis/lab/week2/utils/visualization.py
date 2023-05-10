import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualizeTask1(csv_path):
    "dict: keys the value of alpha and for values [mIoU, mAP]"
    # create the output path : output_path + '/GridSearch' + time
    dict = {}

    # create dictionary form csv
    with open('task1_2.csv') as file:
        reader = csv.reader(file)
        next(reader)  # skip the header row
        for row in reader:
            dict[float(row[0])] = (float(row[1]), float(row[2]))

    # plot mIoU scatter plot
    plt.figure()
    plt.scatter(list(dict.keys()), [x[1] for x in dict.values()], label='IoU')
    plt.xlabel('alpha')
    plt.ylabel('IoU')
    plt.title('IoU vs alpha')
    plt.xticks(list(dict.keys()))
    plt.savefig('IoU_task1.png')

    # plot mAP
    plt.figure()
    plt.scatter(list(dict.keys()), [x[0] for x in dict.values()], label='mAP')
    plt.xlabel('alpha')
    plt.ylabel('mAP')
    plt.title('mAP vs alpha')
    plt.xticks(list(dict.keys()))
    plt.savefig('task1_mAP.png')

    # the two metrics together
    plt.figure()
    plt.scatter(list(dict.keys()), [x[0] for x in dict.values()], label='mAP', color='orange')
    plt.scatter(list(dict.keys()), [x[1] for x in dict.values()], label='IoU', color='blue')
    plt.xlabel('alpha')
    plt.ylabel("metric's value")
    plt.title('mAP and IoU vs alpha')
    plt.xticks(list(dict.keys()))  # set x-axis ticks to dictionary keys
    plt.legend()
    plt.savefig('task1_mAP_IoU.png')


def visualizeTask2(dict):
    """dict: keys the value of alpha and rho and for values [mIoU, mAP]
     dic[alpha][rho] = [map,iou]"""

    # extract the alpha, rho, and value data from the results dictionary
    alphas = []
    rhos = []
    valuesMAP = []
    valuesIoU = []
    for key, value in dict.items():
        for key2, value2 in value.items():
            alphas.append(float(key))
            rhos.append(float(key2))
            valuesMAP.append(float(value2[0]))
            valuesIoU.append(float(value2[1]))

    # create a 3D scatter plot for each alpha and rho values plot the mAP
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(alphas, rhos, valuesMAP)

    # set axis labels and title
    ax.set_xlabel('alpha')
    ax.set_ylabel('rho')
    ax.set_zlabel('mAP')
    ax.set_title('Grid Search Results for the mAP')
    plt.savefig('task2_mAP.png')

    # plot mIoU scatter plot in 3D, for each alpha the rho values
    # create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(alphas, rhos, valuesIoU)

    # set axis labels and title
    ax.set_xlabel('alpha')
    ax.set_ylabel('rho')
    ax.set_zlabel('IoU')
    ax.set_title('Grid Search Results for the IoU')
    plt.savefig('task2_IoU.png')

    # surface plots
    # create 3D surface plots for mAP and IoU
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    X = np.array(alphas).reshape(-1, len(set(rhos)))
    Y = np.array(rhos).reshape(-1, len(set(rhos)))
    Z1 = np.array(valuesMAP).reshape(X.shape)
    Z2 = np.array(valuesIoU).reshape(X.shape)

    ax1.plot_surface(X, Y, Z1, cmap='coolwarm')
    ax2.plot_surface(X, Y, Z2, cmap='coolwarm')

    # set axis labels and titles
    ax1.set_xlabel('alpha')
    ax1.set_ylabel('rho')
    ax1.set_zlabel('mAP')
    ax1.set_title('mAP')

    ax2.set_xlabel('alpha')
    ax2.set_ylabel('rho')
    ax2.set_zlabel('IoU')
    ax2.set_title('IoU')
    # add colorbar
    fig.colorbar(ax1.plot_surface(X, Y, Z1, cmap='coolwarm'), ax=ax1)
    fig.colorbar(ax2.plot_surface(X, Y, Z2, cmap='coolwarm'), ax=ax2)

    plt.tight_layout()
    plt.savefig('task2_surfaces.png')

    # surfaces

    # create a table with pandas and save it as a csv file to save the mAP and IoU for each alpha and rho
    # save the dictionary of dictionaries as a csv file
    df = pd.DataFrame()

    # append columns
    df['alpha'] = alphas
    df['rho'] = rhos
    df['mAP'] = valuesMAP
    df['IoU'] = valuesIoU
    df = df[['alpha', 'rho', 'mAP', 'IoU']]

    df.to_csv('task2.csv')


if __name__ == '__main__':
    csv = 'task1_2.csv'

    visualizeTask1(csv)

    dict = {3.0: {0.025: [0.6095715757213056, 0.33998388822315234], 0.05: [0.409854122374424, 0.33491007433142356],
                  0.075: [0.33634301100046, 0.3256730982034141], 0.1: [0.2955806056862008, 0.3233136156148798]},
            4.0: {0.025: [0.7844494960260728, 0.35072894059941945], 0.05: [0.7703552605821228, 0.34009159852297494],
                  0.075: [0.6064042531866529, 0.3364813617388695], 0.1: [0.6006395668035182, 0.339446408367616]},
            5.0: {0.025: [0.4930055224941774, 0.27965897449308125], 0.05: [0.4991056274097067, 0.2718995737835347],
                  0.075: [0.45341478006024405, 0.26962992101208877], 0.1: [0.3742725945432763, 0.27176920599534354]},
            6.0: {0.025: [0.31867243956197555, 0.20037471172752763], 0.05: [0.31544280254701196, 0.20172187823544374],
                  0.075: [0.3010586864622833, 0.19850633067135176], 0.1: [0.29327115004001125, 0.1975746203489469]},
            7.0: {0.025: [0.16076639109925267, 0.12432141542662974], 0.05: [0.160815382058935, 0.13984700991198146],
                  0.075: [0.16516247446525614, 0.13993597365750685], 0.1: [0.16609114480306228, 0.14474858892491055]}}

    visualizeTask2()
