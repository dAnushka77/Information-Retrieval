import os                           # For file operations and file paths
import numpy as np                  # For numerical operations


def reading_wts_files(folderpath):
    weights_dict = {}                    # Initializing an empty dictionary to store weights of the words
    try:
        for filename in os.listdir(folderpath):    # Looping through the .wts files in the specified folder
            if filename.endswith(".wts"):           # To verify if the files have a .wts extension
                file_path = os.path.join(folderpath, filename)     # For full file path
                with open(file_path, 'r') as file:                  # Opening file in 'read' mode to read file contents
                    line = file.readline().strip()  # Read file's first line and remove whitespace
                    parts = line.split(': ')        # Splitting file lines with ':' as the delimiter
                    if len(parts) == 2:             # Checking if the file line is split into exactly 2 parts
                        # Extracting the tf-idf weights and saving them into the 'weights' dictionary
                        word, tfidf = parts[0].split(':')[-1].strip(), float(parts[1].split('-')[-1].strip())
                        weights_dict[word] = tfidf
                    else:
                        print(f"Skipping file {filename} due to: Invalid format")  # If file format is invalid
        return weights_dict
    except Exception as e:
        print("Error while reading .wts files:", e)
        return None         # Indicates error or failure


def computing_similarity_matrix(weights_dict):
    try:
        words = list(weights_dict.keys())        # Getting a list of words from the dictionary
        num_of_words = len(words)              # Getting the length of the list of words from the 'weights' dictionary
        similarity_matrix = np.zeros((num_of_words, num_of_words))    # Initialize a matrix of zeros to store the computed similarity matrix

        for i in range(num_of_words):          # Iterating over each word in the index
            for j in range(i, num_of_words):   # Such an iteration prevents the double computation of cosine similarity
                similarity = cosine_similarity(weights_dict, words[i], words[j])     # Calculating cosine similarity between 2 words
                similarity_matrix[i, j] = similarity    # Assigning similarity value to the upper triangular part of the matrix
                similarity_matrix[j, i] = similarity    # Since similarity is symmetric, assign similarity value to the lower triangular part of the matrix
        return similarity_matrix
    except Exception as e:
        print("Error while computing the similarity matrix:", e)
        return None         # Indicates error or failure


def cosine_similarity(weights, word1, word2):
    try:
        tfidf1 = weights.get(word1, 0)  # Getting the tf-idf weight of word1 from the 'weights' dictionary. If not found, default = 0. 
        tfidf2 = weights.get(word2, 0)  # Getting the tf-idf weight of word2 from the 'weights' dictionary. If not found, default = 0.
        dot_prod = tfidf1 * tfidf2   # Calculating the dot product of the tf-idf weights calculated above
        norm_word1 = tfidf1 ** 2        # Squared normalized weight of word1
        norm_word2 = tfidf2 ** 2        # Squared normalized weight of word2
        
        if norm_word1 == 0 or norm_word2 == 0:      # Checking if any normalized values are 0 (prevents division by 0 error)
            return 0
        else:
            return dot_prod / (norm_word1 * norm_word2) ** 0.5   # Calculating the cosine similarity
    except Exception as e:
        print("Error while computing cosine similarity:", e)
        return None         # Indicates error or failure


def algo_hierarchical_clustering(similarity_matrix, threshold=0.4):
    try:
        num_of_words = similarity_matrix.shape[0]          # Getting the number of words from the similarity matrix
        clusters = {i: [i] for i in range(num_of_words)}   # Initializing each word as a cluster
        first_100_merges = []                           # To store the merges
        
        while len(clusters) > 1:
            min_dist = np.inf                           # Initializing the minimum distance to infinity
            merge_indices = None                        # Initializing the merge inidices variable
            for i in clusters:
                for j in clusters:
                    if i != j:                          # Ensuring the clusters are different
                        # Calculating the average linkage distance between cluster i and cluster j
                        dist = cluster_distance_avg_linkage(clusters[i], clusters[j], similarity_matrix)
                        if dist < min_dist:             # Checking if calculated distance is less than the current minimum distance
                            min_dist = dist             # Then update the minimum distance to the claculated distance
                            merge_indices = (i, j)      # Updated the merges indices
            if min_dist <= threshold:                   # Break the loop if the minimum distance is less than the threshold value
                break
            
            cluster1, cluster2 = merge_indices          # Merging the clusters having the smallest distance between them
            clusters[cluster1].extend(clusters[cluster2])   # Appending the cluster2 elements to cluster1
            del clusters[cluster2]                      # Then deleting cluster2 from the 'cluster' dictionary
            
            # Calculating the cosine similarity score for the merged clusters
            merge_similarity = similarity_matrix[cluster1, cluster2]

            # Appending the merge indices and cosine similarity scores to the first 100 merges list
            first_100_merges.append((cluster1, cluster2, merge_similarity))
        return first_100_merges
    
    except Exception as e:
        print("Error in hierarchical clustering:", e)
        return None             # Indicates error or failure


def cluster_distance_avg_linkage(cluster1, cluster2, similarity_matrix):
    total_similarity = 0               # To store the total similarity between clusters
    for i in cluster1:          # Iterating over each index in cluster1
        for j in cluster2:      # Iterating over each index in cluster2
            total_similarity += similarity_matrix[i, j]            # Adding the similarity value between i and j to the total similarity value
    return total_similarity / (len(cluster1) * len(cluster2))      # Calculating the avg. similarity value between cluster 1 and cluster 2 and returning it


def print_merges(first_100_merges, most_similar_pair,most_dissimilar_pair):
    # Printing the most similar and most dissimilar pairs of documents
    print("The most similar pair of documents is:", most_similar_pair)
    print("The most dissimilar pair of documents is:", most_dissimilar_pair)
    for i, merge_data in enumerate(first_100_merges):
        # Checking the length format of merge_data list
        if len(merge_data) == 3:        # if length = 3 then it includes similarity data
            cluster1, cluster2, similarity = merge_data     # Unpacking merge_data
            print(f"Merge {i+1}: Cluster {cluster1} and Cluster {cluster2}, Similarity: {similarity}")
        elif len(merge_data) == 2:      # if length = 2 then it does not includes similarity data
            cluster1, cluster2 = merge_data                 # Unpacking merge_data
            print(f"Merge {i+1}: Cluster {cluster1} and Cluster {cluster2}")
        else:                           # if length not equal to 2 or 3, then it is an invalid data format
            print(f"Merge {i+1}: Invalid merge data format")


def most_similar_dissimilar_pairs(similarity_matrix):
    try:
        num_of_docs = similarity_matrix.shape[0]   # Getting the number of rows (docs) in the similarity matrix
        most_similar_pair = ()                  # Initializing tuple for most similar pair of docs
        most_dissimilar_pair = ()               # Initializing tuple for most dissimilar pair of docs
        max_sim = 0                             # Initializing maximum similarity to 0
        min_sim = 1                             # Initializing minimum similarity to 1 (cosine similarity lies between -1 and 1, both inclusive)

        for i in range(num_of_docs):               # Iterating over all document pairs
            for j in range(i + 1, num_of_docs):
                sim = similarity_matrix[i, j]   # Getting the similarity score between document i and j
                # Updating the most similar pair and maximum similarity if the similarity score value is greater than the current maximum similarity
                if sim > max_sim:
                    max_sim = sim
                    most_similar_pair = (i, j)
                # Updating the most dissimilar pair and minimum similarity if the similarity score value is smaller than the current maximum similarity
                if sim < min_sim:
                    min_sim = sim
                    most_dissimilar_pair = (i, j)

        return most_similar_pair, most_dissimilar_pair
    except Exception as e:
        print("Error in finding the most similar and most dissimilar pairs of documents:", e)
        return None, None       # If error or failure is found, the function returns None for both - most similar pair and most dissimilar pair of documents


def closest_to_centroid(similarity_matrix):
    try:
        num_docs = similarity_matrix.shape[0]   # Fetting the number of rows (i.e. docs) in the similarity matrix
        centroid_similarity = []                # Empty list to store the avg. similarity of each doc to all other docs
        for i in range(num_docs):
            # Calculating the avg. similarity of each doc (i) to all other docs
            avg_similarity = similarity_matrix[i, :].mean()
            # Appending avg. similarity to the list
            centroid_similarity.append(avg_similarity)

        # Finding the index of the doc with the highest avg. similarity
        closest_doc_index = np.argmax(centroid_similarity)
        return closest_doc_index
    except Exception as e:
        print("Error in finding the document closest to the centroid:", e)
        return None                             # Indicates error or failure


def main(folderpath):
    try:
        # Reading the .wts files
        weights = reading_wts_files(folderpath)    
        if weights is None:
            return
        print("All files read successfully!")
        
        # Computing the similarity matrix using the weights
        similarity_matrix = computing_similarity_matrix(weights)    
        if similarity_matrix is None:
            return
        print("Similarity matrix computed successfully!")

        threshold = 0.4                  # Setting a threshold value for the hierarchical clustering algorithm
        
        # Performing hierarchical clustering
        first_100_merges = algo_hierarchical_clustering(similarity_matrix, threshold)       
        if first_100_merges is None:
            return

        # Finding the most similar and most dissimilar pairs of docs
        most_similar_pair, most_dissimilar_pair = most_similar_dissimilar_pairs(similarity_matrix)
        if most_similar_pair is None or most_dissimilar_pair is None:
            return
        
        # Printing the cluster merges and the similar and dissimilar document pairs
        print_merges(first_100_merges, most_similar_pair, most_dissimilar_pair)

        # Finding the document closest to the centroid
        closest_doc_index = closest_to_centroid(similarity_matrix)
        if closest_doc_index is not None:
            print("The document which is closest to the centroid is:", closest_doc_index)
    
    except Exception as e:
        print("Error:", e)

# Defining the folder path which contains .wts files
folderpath = "D:/SEM_4/Project/phase5/final_submission/output_files"
main(folderpath)