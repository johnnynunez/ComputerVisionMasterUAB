import os
import matplotlib.pyplot as plt
import numpy as np


def extract_retrieval_examples_img2text(args,neighbors_index, neighbors_id, databaseDataset, queryDataset, output_path, distances=None):
    
    query_list = [0, 1, 2, 3, 4, 5, 6, 7]

    images = []
    captions = []
    
    for query in query_list:
        path = os.path.join(output_path, f'query_{query}')
        # Create directory if it does not exist
        os.makedirs(path, exist_ok=True)

        print(query)
        print("Query image:")
        # Get image
        img, _ = queryDataset[query]
        img = np.array(img).transpose(1,2,0)
        plt.imshow(img)
        plt.savefig(os.path.join(path, 'query.png'))

        images.append(img)
        
        # Get Captions
        captionIds = queryDataset.getCaptionsId_fromImageIdx(query)
        captionStr = [databaseDataset.getstrCaption_fromCaptionIdx(databaseDataset.getCaptionIdx_fromId(capt)) for capt in captionIds]
        print("Query Captions GT: ", captionStr)
        
        # Write text file with objects
        with open(os.path.join(path, 'query.txt'), 'w') as f:
            f.write("Query Captions GT: ")
            for capt in captionStr:
                f.write(capt[0])
                f.write(" \n")
        
        img_captions = []

        # Get 5 most close strings
        for i in range(5): 
            neighbor_id = neighbors_id[query, i]
            neighbor_idx = neighbors_index[query, i]
            
            # Get caption and it's correspondent image
            caption = databaseDataset.getstrCaption_fromCaptionIdx(neighbor_idx)
            print(i, ". Closest caption:", caption)
          
            
            if distances is not None:
                print("Caption (at distance " + str(round(distances[query, i], 4)) + "):", caption)

            img_captions.append(caption)
            
            imageDB_id = databaseDataset.getImageId_fromCaptionIdx(neighbor_idx)[0]
            imageDB_idx = queryDataset.getImageIdx_fromId(imageDB_id)
            imageDB, imageDB_id = queryDataset[imageDB_idx]
            img = np.array(imageDB).transpose(1,2,0)
            plt.imshow(img)
            plt.savefig(os.path.join(path, f'neighbor_{i}.png'))
            
            if distances is not None:
                with open(os.path.join(path, 'query.txt'), 'a') as f:
                    f.write(f'neighbor_{i} Caption (at distance {str(round(distances[query, i], 4))}): {caption}\n')
            else:
                with open(os.path.join(path, 'query.txt'), 'a') as f:
                    f.write(f'neighbor_{i} Caption: {caption}\n')
            
            
        captions.append(img_captions)

    # plot_retrieval(images, captions, output_path)
    
    
    
def extract_retrieval_examples_txt2img(args, neighbors_index, neighbors_id, databaseDataset, queryDataset, output_path, distances=None):
    
    query_list = [0, 1, 2, 3, 4, 5]

    images = []
    captions = []
    
    for query in query_list:
        path = os.path.join(output_path, f'query_{query}')
        # Create directory if it does not exist
        os.makedirs(path, exist_ok=True)

        print(query)
        print("Query Caption:")
        # Get Caption
        _, caption_id = queryDataset[query]
        caption = queryDataset.getstrCaption_fromCaptionIdx(query)
        print(caption)
        # Write text file with objects
        with open(os.path.join(path, 'query.txt'), 'w') as f:
            f.write("Query Caption: ")
            f.write(caption)
            f.write(" \n")
            
        # Get GT image
        image_id = queryDataset.getImageId_fromCaptionId(caption_id)[0]
        image, image_id = databaseDataset[databaseDataset.getImageIdx_fromId(image_id)]
        img = np.array(image).transpose(1,2,0)
        plt.imshow(img)
        plt.savefig(os.path.join(path, f'GT_image.png'))
        
        
        # Get 5 most close strings
        for i in range(5):
            print(i, ". closest image:")
            
            neighbor_id = neighbors_id[query, i]
            neighbor_idx = neighbors_index[query, i]
            
            image, image_id = databaseDataset[neighbor_idx]
            
            img = np.array(image).transpose(1,2,0)
            plt.imshow(img)
            plt.savefig(os.path.join(path, f'neighbor_{i}.png'))
    
            
            # if distances is not None:
            #     print("Image (at distance " + str(round(distances[query, i], 4)) + "):", caption)
            #     with open(os.path.join(path, 'query.txt'), 'a') as f:
            #         f.write(f'neighbor_{i} Image (at distance {str(round(distances[query, i], 4))}): {caption}\n')
            # else:
            #     print("Image:", caption)
            #     with open(os.path.join(path, 'query.txt'), 'a') as f:
            #         f.write(f'neighbor_{i} Image: {caption}\n')


    # plot_retrieval(images, captions, output_path)


def plot_retrieval(images, captions, output_path, num_queries=3, num_retrievals=5):

    title = "First test query images and their retrieved captions\n"

    fig, ax = plt.subplots(num_queries, 2, figsize=(20, 10), dpi=200)
    fig.suptitle(title)
    for i in range(0, num_queries):
        ax[i][0].imshow(images[i])
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])
        ax[i][0].set_title("Query image")
        ax[i][1].set_title("Retrieved captions")
        for j in range(0, num_retrievals):
            ax[i][1].text(0.05, 0.9-j*0.2, "- " + captions[i][j], horizontalalignment='left',
                         verticalalignment='center', fontsize=7, transform=ax[i][1].transAxes)
            ax[i][1].set_xticks([])
            ax[i][1].set_yticks([])
    fig.tight_layout()
    plt.savefig(output_path + "/RetrievalQualitativeResults.png")
    plt.close()
               
