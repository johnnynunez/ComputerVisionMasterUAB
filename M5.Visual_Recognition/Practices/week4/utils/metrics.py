import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve,accuracy_score
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns
import cv2

# if torch.cuda.is_available():
#     print("CUDA is available")
#     device = torch.device("cuda")
#     torch.cuda.amp.GradScaler()
# elif torch.backends.mps.is_available():
#     print("MPS is available")
#     device = torch.device("mps")
# else:
#     print("CPU is available")
#     device = torch.device("cpu")


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def tsne_features(image_features, y_true, labels, title, output_path):
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(image_features, y_true)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink']
    for i, c in enumerate(set(y_true)):
        mask = y_true == c
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=labels[i], c=colors[i], alpha=0.7)
    ax.legend()
    ax.set_title(title)
    fig.savefig(output_path)

    
def plot_retrieval(
        test_images,
        train_images,
        y_true_test,
        y_true_train,
        neigh_ind,
        neigh_dist,
        output_dir,
        p="BEST",
        num_queries=8,
        num_retrievals=5,
):
    if p == "BEST":
        ind = np.argsort(np.sum(neigh_dist[:, 0:5], axis=1), axis=0)
        title = "Test query images that obtained retrieved images \nwith the lowest distance among the dataset\n"
    elif p == "WORST":
        ind = np.argsort(-np.sum(neigh_dist[:, 0:5], axis=1), axis=0)
        title = "Test query images that obtained retrieved images \nwith the highest distance among the dataset\n"
    else:
        # get one image from each class
        ind = []
        for i in range(0, 8):
            ind.append(np.where(y_true_test == i)[0][0])

        title = "One test query image fom each class and their retrieved train images\n"

  
    test_images = test_images[ind]
    y_true_test = y_true_test[ind]
    neigh_ind = neigh_ind[ind]
    neigh_dist = neigh_dist[ind]

    fig, ax = plt.subplots(num_queries, num_retrievals, figsize=(10, 15), dpi=200)
    fig.suptitle(title)
    for i in range(0, num_queries):
        #normalize the images so that they can be plotted in RGB
        # find the min and max values of the image
        min_val = np.min(test_images[i])
        max_val = np.max(test_images[i])
        # normalize the image
        test_images[i] = (test_images[i] - min_val) / (max_val - min_val)

        ax[i][0].imshow(np.moveaxis(test_images[i], 0, -1))
        ax[i][0].set_title("Query image \nClass: " + str(y_true_test[i]))
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])
        for j in range(1, num_retrievals):
            # find the min and max values of the image
            min_val = np.min(train_images[neigh_ind[i][j]])
            max_val = np.max(train_images[neigh_ind[i][j]])
            # normalize the image
            train_images[neigh_ind[i][j]] = (train_images[neigh_ind[i][j]] - min_val) / (max_val - min_val)

            ax[i][j].imshow(np.moveaxis(train_images[neigh_ind[i][j]], 0, -1))
            ax[i][j].set_title("Retrived image " + str(j))
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
    fig.tight_layout()
    plt.savefig(output_dir + "/ImageRetrievalQualitativeResults_" + p + ".png")
    plt.close()

def plot_retrieval_2(
        test_images,
        train_images,
        y_true_test,
        y_true_train,
        neigh_ind,
        neigh_dist,
        output_dir,
        p="BEST",
        num_queries=8,
        num_retrievals=5,
):
    labelsName = np.array(
        ['Opencountry', 'coast', 'forest', 'highway', 'inside_city', 'mountain', 'street', 'tallbuilding'])
    labelsDict = {index: label for index, label in enumerate(labelsName)}
    if p == "BEST":
        ind = np.argsort(np.sum(neigh_dist[:, 0:5], axis=1), axis=0)
        title = "Test query images that obtained retrieved images \nwith the lowest distance among the dataset\n"
    elif p == "WORST":
        ind = np.argsort(-np.sum(neigh_dist[:, 0:5], axis=1), axis=0)
        title = "Test query images that obtained retrieved images \nwith the highest distance among the dataset\n"
    else:
        # get one image from each class
        ind = []
        for i in range(0, 8):
            ind.append(np.where(y_true_test == i)[0][0])
        ind = np.array(ind)

        title = "One test query image fom each class and their retrieved train images\n"

    test_images = test_images[ind]
    y_true_test = y_true_test[ind]
    neigh_ind = neigh_ind[ind]
    neigh_dist = neigh_dist[ind]

    fig, ax = plt.subplots(num_queries, num_retrievals, figsize=(10, 15), dpi=200)
    fig.suptitle(title)
    for i in range(0, num_queries):
        # normalize the images so that they can be plotted in RGB
        # find the min and max values of the image
        min_val = np.min(test_images[i])
        max_val = np.max(test_images[i])
        # normalize the image
        test_images[i] = (test_images[i] - min_val) / (max_val - min_val)

        ax[i][0].imshow(np.moveaxis(test_images[i], 0, -1))
        ax[i][0].set_title("Query image \nClass: " + labelsDict[i])
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])
        for j in range(0, num_retrievals):
            # find the min and max values of the image
            min_val = np.min(train_images[neigh_ind[i][j]])
            max_val = np.max(train_images[neigh_ind[i][j]])
            # normalize the image
            train_images[neigh_ind[i][j]] = (train_images[neigh_ind[i][j]] - min_val) / (max_val - min_val)

            ax[i][j].imshow(np.moveaxis(train_images[neigh_ind[i][j]], 0, -1))
            ax[i][j].set_title("Retrived image " + str(j) + "\nClass: " + labelsDict[y_true_train[neigh_ind[i][j]]])
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
    fig.tight_layout()
    plt.savefig(output_dir + "/ImageRetrievalQualitativeResults_" + p + ".png")
    plt.close()

def plot_embeddings(embeddings, targets, classes, title, output_path, xlim=None, ylim=None):
    colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink']
    plt.figure(figsize=(10, 10))
    for i in range(len(classes)):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()
    
def plot_embeddings_coco(embeddings, target, classes, title, output_path, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
 
    for i in range(len(embeddings)):
        plt.scatter(embeddings[i,0], embeddings[i,1], alpha=0.5)

    plt.scatter(embeddings[:,0], embeddings[:,1], alpha=0.5)
    # Adjust the plot limits (just to make sure the point cloud is fully visible)
    
    x_min, x_max = embeddings[:,0].min(), embeddings[:,0].max()
    y_min, y_max = embeddings[:,1].min(), embeddings[:,1].max()
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    # plt.legend(classes)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()


def extract_embeddings(dataloader, model, device):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if torch.cuda.is_available():
                images = images.to(device)
            embeddings[k:k + len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k + len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

def extract_embeddings_coco(dataloader, model, device):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 1024))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if torch.cuda.is_available():
                images = images.to(device)
            embeddings[k:k + images.shape[0]] = model.get_embedding(images).data.cpu().numpy()
            # labels[k:k + len(images)] = target.numpy()
            k += images.shape[0]
    return embeddings



def plot_PR_multiclass (classes, labels, y_score_knn,path):
    precision = dict()
    recall = dict()
    for i in range(len(classes)):
        labels_val = np.where(labels == i, 1, 0)
        ap = round(average_precision_score(labels_val,  y_score_knn[:, i]),2)
        #append the average precision for each class rounded to 2 decimals
        precision[i], recall[i], _ = precision_recall_curve(labels_val, y_score_knn[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='{} {},'.format(classes[i], ap))
        
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")

    plt.savefig(path + "/PrecisionRecallCurve.png")
    plt.close()
    
def plot_PR_binary(results, path):
    precision, recall = precisionRecall(results) 
    plt.figure(figsize=(10, 10))
    plt.plot(recall, precision, lw=2)
    # Set the x and y axis limits accordingly to the data
    
    x_min, x_max = 0, 1
    y_min , y_max = 0, 1
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.savefig(path + "/PrecisionRecallCurve.png")
    plt.close()


def calculate_APs (y_true_test, y_true_test_repeated, neigh_labels, neigh_dist):
    for k in [1, 3, 5, 10, 20, 30]:
        prec_at_k = accuracy_score(y_true_test_repeated[:, 0:k].flatten(), neigh_labels[:, 0:k].flatten())
        print("Prec@" + str(k) + ":", prec_at_k)

    aps_all = []
    aps_5 = []
    for i in range(neigh_labels.shape[0]):
        aps_all.append(average_precision_score((neigh_labels[i] == y_true_test[i]).astype(int), -neigh_dist[i]))
        aps_5.append(average_precision_score((neigh_labels[i, 0:5] == y_true_test[i]).astype(int), -neigh_dist[i, 0:5]))
    mAP_all = np.mean(aps_all)
    mAP_5 = np.mean(aps_5)

    print("mAP@all:", mAP_all)
    print("mAP@5:", mAP_5)
    
    
def calculate_APs_coco (results, path):
    results_txt = []
    for k in [1, 3, 5]:
        prec_at_k = mPrecisionK(results, k)
        print("Prec@" + str(k) + ":", prec_at_k)
        results_txt.append("Prec@" + str(k) + ": " + str(prec_at_k))

    print("mAP:", MAP(results))
    results_txt.append("mAP: " + str(MAP(results)))
    
    # Save results in .txt file
    with open(path + "/results.txt", "w") as output:
        output.write(str(results_txt))
    
    
    
def positive_image(objs1, objs2):
        
        for obj in objs1:
            if obj in objs2:
                return True
        return False
    
def positives_coco(neighbors, databaseDataset, queryDataset):
    
    resultsQueries = []
    
    for i_query in tqdm(range(neighbors.shape[0])):
        resultQuery = []
        
        queryObjs = queryDataset.getObjs(i_query)
        
        for i_db in range(neighbors.shape[1]):
            
            dbIndex = neighbors[i_query, i_db]
            
            dbObjs = databaseDataset.getObjs(dbIndex)
            
            if positive_image(dbObjs, queryObjs): 
                resultQuery.append(1)
            else:
                resultQuery.append(0)
        
        resultsQueries.append(resultQuery)
    
    return np.array(resultsQueries)


def precisionK(results, k):
    """
    This function computes the precision@k for a query
    giving the positive results
    Parameters
    ----------
    results : numpy array
        Array with 1 when retrive was positive, 0 otherwise.
    k : int
        k value to compute.
    Returns
    -------
    float
        p@k value.
    """
    
    return np.sum(results[:k])/k

def mPrecisionK(listResults, k):
    """
    This function computes the mean precision@k over all the queries.
    Parameters
    ----------
    listResults : numpy array
        For each query (row), 1 if retrieve was positive 0 otherwise.
    k : int
        k value to compute.
    Returns
    -------
    float
        Mean p@k value.
    """
    
    valSum = 0
    
    for i in range(listResults.shape[0]):
        valSum += precisionK(listResults[i,:], k)
    
    return valSum / listResults.shape[0]

def recallK(results, k):
    """
    This function computes the recall@k for a query
    giving the positive results
    Parameters
    ----------
    results : numpy array
        Array with 1 when retrive was positive, 0 otherwise.
    k : int
        k value to compute.
    Returns
    -------
    float
        r@k value.
    """
    
    return np.sum(results[:k])/np.sum(results)

def mRecallK(listResults, k):
    """
    This function computes the mean recall@k over all the queries.
    Parameters
    ----------
    listResults : numpy array
        For each query (row), 1 if retrieve was positive 0 otherwise.
    k : int
        k value to compute.
    Returns
    -------
    float
        Mean r@k value.
    """
    
    
    valSum = 0
    
    for i in range(listResults.shape[0]):
        valSum += recallK(listResults[i,:], k)
    
    return valSum / listResults.shape[0]
    
def averagePrecision(results):
    """
    This function computes the average precision for a query
    giving the positive results
    Parameters
    ----------
    results : numpy array
        Array with 1 when retrive was positive, 0 otherwise.
    Returns
    -------
    float
        ap value.
    """
    
    
    ap = (np.cumsum(results) * results)/(np.array(range(results.shape[0])) + 1)
    
    if np.sum(results) == 0:
        return 0
    
    return np.sum(ap)/np.sum(results)

def MAP(listResults):
    """
    This function computes the mean average previcision over all the queries.
    Parameters
    ----------
    listResults : numpy array
        For each query (row), 1 if retrieve was positive 0 otherwise.
    Returns
    -------
    float
        Mean ap value.
    """
    
    valSum = 0
    
    for i in range(listResults.shape[0]):
        valSum += averagePrecision(listResults[i,:])
    
    return valSum / listResults.shape[0]

def precisionRecall(listResults):
    """
    This function computes the mean precision and recall of all queries.
    Parameters
    ----------
    listResults : numpy array
        For each query (row), 1 if retrieve was positive 0 otherwise.
    Returns
    -------
    numpy array, numpy array
        Mean precision and recall values
    """
    values = (np.array(range(listResults.shape[1])) + 1)
    values = values[np.newaxis, ...]
    p = np.cumsum(listResults, axis=1)/values
    positiveSum = (np.sum(listResults,axis = 1))
    positiveSum = positiveSum[:,np.newaxis]+1e-10
    r = np.cumsum(listResults, axis=1)/positiveSum 
    
                                    
    mp = np.mean(p, axis=0)
    mr = np.mean(r, axis=0)

    
    return mp, mr
