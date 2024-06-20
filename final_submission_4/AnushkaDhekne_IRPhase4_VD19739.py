import os                   # Importing the os library for the file operations
import argparse             # Importing the argparse library for parsing command-line args

# Reading the inverted index and postings
def reading_index_and_postings(index_dir):
    dictionary_file = os.path.join(index_dir, "dictionary.txt")         # Constructing file path for the dictionary.txt file
    postings_file = os.path.join(index_dir, "postings.txt")             # Contructing the file path for the postings.txt file
    inverted_index = {}                                                 # Initializing an empty dictionary to store the inverted index
    with open(dictionary_file, 'r', encoding='utf-8') as dict1:         # Open and read the dictionary.txt file
        for line in dict1:                                              # Iterating over the dictionary.txt file's lines
            each_term = line.strip()                                    # Extracting each word from the line and removing white spaces
            num_of_docs = int(next(dict1).strip())                      # Reading the number of documents containing the term
            starting_position = int(next(dict1).strip())                # Reading the starting position of the postings for each of the terms in the dictionary.txt file
            # Storing the number of documents, starting position and the term in the inverted index dictionary
            inverted_index[each_term] = {'number_of_docs': num_of_docs, 'starting_position': starting_position}
    return inverted_index, postings_file                                # Returning the inverted index dict and the path to the postings file

# Reading the postings for a given term
def reading_postings(each_term, postings_file, num_of_docs, starting_position):
    postings_dict = {}                                                  # Initializing an empty dict to store doc IDs and term weights
    with open(postings_file, 'r', encoding='utf-8') as file:            # Open and read the postings.txt file
        file.seek(starting_position)                                    # Assigning the file pointer to the posting file's start position for a given term
        for _ in range(num_of_docs):                                    # Iterating over the number of docs containing the term
            lines = file.readline().strip()                             # Extracting each word from the line and removing white spaces
            if ',' not in lines:                                        # Checking if a line contains a comma (this indicates doc ID and weight)
                continue
            document_id, weights = lines.split(',', 1)                  # Split the line into doc ID and weight - split only once to handle cases where weight may contain commas
            postings_dict[document_id.strip()] = float(weights.strip()) # Store the doc ID and weights in the postings_dict dictionary
    return postings_dict

# Preprocessing the queries and extracting the term weights
def preprocessing_queries(queries):
    query_terms = []                                  # Empty list to store the query terms
    query_weights = []                                # Empty list to store query weights
    i = 0
    while i < len(queries):                           # Iterating through the list of the queries
        if queries[i].replace('.', '', 1).isdigit():  # Checking if 'i' is a weight
            query_weights.append(float(queries[i]))   # Converting the element 'i' to float and appending it to the query_weights list
            i += 1
        else:
            query_terms.append(queries[i].lower())    # Converting the element 'i' to lowercase and appending it to the query_terms list
            i += 1
    return query_terms, query_weights

# Retrieving document weights based on the command-line queries
def doc_retrieval(query_terms, query_weights, inverted_index, postings_file):
    doc_weights = {}                                        # Initializing an empty dict for storing doc weights
    for term, weight in zip(query_terms, query_weights):    # Iterating through query_terms and query_weights lists
        if term in inverted_index:                          # Checking if the term exists in the inverted index dict
            postings_info = inverted_index[term]            # Getting the postings info for the term in the dict
            # Reading the postings for each of the temrs in the postings.txt file
            postings = reading_postings(term, postings_file, postings_info['number_of_docs'], postings_info['starting_position'])
            for doc_id, doc_weight in postings.items():     # Iterating through the postings 
                # Initializing doc ID if it is not present in the doc_weights dict
                if doc_id not in doc_weights:
                    doc_weights[doc_id] = 0
                doc_weights[doc_id] += doc_weight * weight  # Applying the query term weight to the doc_weight
    doc_weights.pop('', None)                               # Removing the initial empty string from the document weights dict
    return doc_weights

# Displaying the top-10 documents
def top_docs(doc_weights, top=10):
    sorted_docs = sorted(doc_weights.items(), key=lambda x: x[1], reverse=True)             # Sorting the doc_weights dict in descending order by its values 
    non_zero_scores = [(doc_id, weight) for doc_id, weight in sorted_docs if weight > 0]    # Filtering docs with weight equal to zero
    if not non_zero_scores:
        print("No documents with non-zero scores were found for the given query.")
        return
    for i, (doc_id, weight) in enumerate(non_zero_scores[:top], start=1):                   # Printing the top 10 docs and their weights
        print(f"{doc_id}.html {weight:.5f}")

def main():
    parser = argparse.ArgumentParser(description="Command Line information retrieval engine")   # Creating an argument parser object along with its description
    parser.add_argument("query", nargs='+', help="Query terms and their weights")               # Adding a positional argument 'query' to accept 1/more query terms with their weights
    args = parser.parse_args()                                                                  # Parsing the command line args
    index_dir = "D:/SEM_4/Project/phase4"                                                       # Path to the dictionary.txt and postings.txt files  
    inverted_index, postings_file = reading_index_and_postings(index_dir)                       # Reading the inverted index and postings file
    query_terms, query_weights = preprocessing_queries(args.query)                              # Preprocessing the query terms and extracting their corresponding weights
    doc_weights = doc_retrieval(query_terms, query_weights, inverted_index, postings_file)      # Retrieving query-based doc weights
    top_docs(doc_weights)                                                                       # Printing top 10 docs

if __name__ == "__main__":
    main()