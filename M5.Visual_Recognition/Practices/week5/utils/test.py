import os
import time
import umap
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


from utils import metrics
from utils.retrieval import extract_retrieval_examples_img2text, extract_retrieval_examples_txt2img



def test(args, model, 
         image_dataset, image_loader, 
         text_dataset, text_loader, 
         output_path, 
         dim_features=None, 
         device=None, 
         wandb=None, 
         retrieval=False):
   
    # --------------------------------- INFERENCE --------------------------------
    if dim_features is None:
        dim_features = args.dim_out_fc
    

    print("Calculating image val database embeddings...")
    start = time.time()
    val_embeddings_image, labels_image = metrics.extract_embeddings_image(image_loader, model, device, dim_features = dim_features) # dim_features = 300)
    end = time.time()
    print("Time to calculate image val database embeddings: ", end - start)

    print("Calculating text val database embeddings...")
    start = time.time()
    val_embeddings_text, labels_text = metrics.extract_embeddings_text(text_loader, model, device, network_text = args.network_text, dim_features = dim_features) # dim_features = 300)
    end = time.time()
    print("Time to calculate text val database embeddings: ", end - start)

   
    # --------------------------------- RETRIEVAL ---------------------------------
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=10) 
    
    
    print("Fitting KNN...")
    start = time.time()
    if args.task == 'task_a':
        knn = knn.fit(val_embeddings_text, labels_text)
    elif args.task == 'task_b':
        knn = knn.fit(val_embeddings_image, labels_image)
    else: 
        raise ValueError("Task not recognized")
    end = time.time()

    print("Calculating KNN...")
    start = time.time()
    if args.task == 'task_a':
        neighbors = knn.kneighbors(val_embeddings_image, return_distance=False)
        # Map the indices of the neighbors matrix to their corresponding 'id' values
        id_neighbors_matrix = np.vectorize(lambda i: labels_text[i])(neighbors)
        
    elif args.task == 'task_b':
        neighbors = knn.kneighbors(val_embeddings_text, return_distance=False)
        id_neighbors_matrix = np.vectorize(lambda i: labels_image[i])(neighbors)
    end = time.time()
    print("Time to calculate KNN: ", end - start)
    
    
    
    # Compute positive and negative values
    if args.task == 'task_a':
        evaluation = metrics.positives_ImageToText(neighbors, id_neighbors_matrix, text_dataset, image_dataset)
    elif args.task == 'task_b':
        print("Metrics for task b PENDING TO BE REVIEWED")
        evaluation = metrics.positives_TextToImage(neighbors, id_neighbors_matrix, image_dataset, text_dataset)
    else:
        raise ValueError("Task not recognized")


    # --------------------------------- METRICS ---------------------------------
    map_value = metrics.calculate_APs_coco(evaluation, output_path, wandb)

    
    # --------------------------------- PLOT ---------------------------------
    if retrieval:
        if args.task == 'task_a': 
            extract_retrieval_examples_img2text(args, neighbors, id_neighbors_matrix, databaseDataset=text_dataset, queryDataset=image_dataset, output_path=output_path)
        elif args.task == 'task_b':
            print("Retrieval for task b PENDING TO BE REVIEWED")
            extract_retrieval_examples_txt2img(args, neighbors, id_neighbors_matrix, databaseDataset=image_dataset, queryDataset=text_dataset, output_path=output_path)
        else:
            raise ValueError("Task not recognized")

        #----------------- PR CURVE -----------------
        # metrics.plot_PR_binary(evaluation, output_path, wandb)
    
        # # ----------------- UMAP -----------------
        reducer = umap.UMAP(random_state=42)
        reducer.fit(val_embeddings_image)
        umap_image_embeddings = reducer.transform(val_embeddings_image)
        umap_text_embeddings = reducer.transform(val_embeddings_text)

        
        metrics.plot_embeddings_ImageText(umap_image_embeddings, umap_text_embeddings, "UMAP embeddings representation", output_path)

    
    return map_value