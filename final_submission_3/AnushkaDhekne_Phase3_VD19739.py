import os                           # For file operations
import re                           # For using regular expressions
import argparse                     # For parsing command-line arguments
from collections import Counter     # For calculating token and frequency counts
import math                         # For math calculations and formulae
import time                         # For calculating elapsed time
import matplotlib.pyplot as plt     # For plotting the graph

# Hardcoding the list of given stopwords
STOPWORDS = {
    "a", "about", "above", "according", "across", "actually", "adj", "after", "afterwards", "again", "against", 
    "all", "almost", "alone", "along", "already", "also", "although", "always", "among", "amongst", "an", "and", 
    "another", "any", "anybody", "anyhow", "anyone", "anything", "anywhere", "are", "area", "areas", "aren't", 
    "around", "as", "ask", "asked", "asking", "asks", "at", "away", "back", "backed", "backing", "backs", "be", 
    "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "began", "begin", 
    "beginning", "behind", "being", "beings", "below", "beside", "besides", "best", "better", "between", "beyond", 
    "big", "bill", "billion", "both", "but", "by", "came", "can", "cannot", "caption", "case", "cases", "certain", 
    "certainly", "clear", "clearly", "co", "come", "could", "couldn't", "d", "did", "didn't", "differ", "different", 
    "differently", "do", "does", "doesn't", "don't", "done", "down", "downed", "downing", "downs", "during", "e", 
    "each", "early", "eg", "eight", "eighty", "either", "else", "elsewhere", "end", "ended", "ending", "ends", "enough", 
    "etc", "even", "evenly", "ever", "every", "everybody", "everyone", "everything", "everywhere", "f", "face", "faces", 
    "fact", "facts", "far", "felt", "few", "fifty", "find", "finds", "first", "five", "for", "former", "formerly", 
    "forty", "found", "four", "from", "further", "furthered", "furthering", "furthers", "g", "gave", "general", 
    "generally", "get", "gets", "give", "given", "gives", "go", "going", "good", "goods", "got", "great", "greater", 
    "greatest", "group", "grouped", "grouping", "groups", "h", "had", "has", "hasn't", "have", "haven't", "having", 
    "he", "he'd", "he'll", "he's", "hence", "her", "here", "here's", "hereafter", "hereby", "herein", "hereupon", 
    "hers", "herself", "high", "higher", "highest", "him", "himself", "his", "how", "however", "hundred", "i", "i'd", 
    "i'll", "i'm", "i've", "ie", "if", "important", "in", "inc", "indeed", "instead", "interest", "interested", 
    "interesting", "interests", "into", "is", "isn't", "it", "it's", "its", "itself", "j", "just", "k", "l", "large", 
    "largely", "last", "later", "latest", "latter", "latterly", "least", "less", "let", "let's", "lets", "like", 
    "likely", "long", "longer", "longest", "m", "made", "make", "makes", "making", "man", "many", "may", "maybe", 
    "me", "meantime", "meanwhile", "member", "members", "men", "might", "million", "miss", "more", "moreover", "most", 
    "mostly", "mr", "mrs", "much", "must", "my", "myself", "n", "namely", "necessary", "need", "needed", "needing", 
    "needs", "neither", "never", 'nevertheless', 'new', 'newer', 'newest', 'next', 'nine', 'ninety', 'no', 'nobody', 'non', 
    'none', 'nonetheless','noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'o', 'of', 
    'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one', 'one\'s', 'only', 'onto', 'open', 'opened', 'opens', 'or',
    'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 
    'over', 'overall', 'own', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 
    'point', 'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 
    'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'recent', 'recently', 'right', 'room', 
    'rooms', 's', 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 
    'seems', 'seven', 'seventy', 'several', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'show', 'showed', 
    'showing', 'shows', 'sides', 'since', 'six', 'sixty', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'somehow', 
    'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'state', 'states', 'still', 'stop', 'such', 'sure', 't', 
    'take', 'taken', 'taking', 'ten', 'than', 'that', 'that\'ll', 'that\'s', 'that\'ve', 'the', 'their', 'them', 'themselves', 
    'then', 'thence', 'there', 'there\'d', 'there\'ll', 'there\'re', 'there\'s', 'there\'ve', 'thereafter', 'thereby', 
    'therefore', 'therein', 'thereupon', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'thing', 'things', 
    'think', 'thinks', 'thirty', 'this', 'those', 'though', 'thought', 'thoughts', 'thousand', 'three', 'through', 'throughout', 
    'thru', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'towards', 'trillion', 'turn', 'turned', 'turning', 
    'turns', 'twenty', 'two', 'u', 'under', 'unless', 'unlike', 'unlikely', 'until', 'up', 'upon', 'us', 'use', 'used', 
    'uses', 'using', 'v', 'very', 'via', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', 'wasn\'t', 'way', 'ways', 'we', 
    'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'well', 'wells', 'were', 'weren\'t', 'what', 'what\'ll', 'what\'s', 'what\'ve', 
    'whatever', 'when', 'whence', 'whenever', 'where', 'where\'s', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 
    'wherever', 'whether', 'which', 'while', 'whither', 'who', 'who\'d', 'who\'ll', 'who\'s', 'whoever', 'whole', 'whom', 
    'whomever', 'whose', 'why', 'will', 'with', 'within', 'without', 'won\'t', 'work', 'worked', 'working', 'works', 'would', 
    'wouldn\'t', 'x', 'y', 'year', 'years', 'yes', 'yet', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'young', 
    'younger', 'youngest', 'your', 'yours', 'yourself', 'yourselves', 'z'
}

def processing_documents(input_dir):
    documents = {}                                              # Dictionary to store document ids and their token counters
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):                           # Processing only the .txt files
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()                              #  Reading file contents
                tokens = re.findall(r'\b\w+\b', text.lower())
                tokens = [token for token in tokens if token not in STOPWORDS]             # Removing the stopwords
                tokens_counter = Counter(tokens)                # Counting the frequency of each token
                match = re.search(r'\d+', filename)             # Extracting the document ID using regular expression
                if match:
                    document_id = match.group()
                    documents[document_id] = tokens_counter     # Storing the token counter for the document
    return documents


def calculate_term_weights(documents):
    term_document_frequency = {}                                # Dictionary to store document frequency of each term
    total_documents = len(documents)                            # Total number of documents
    # Calculating the document frequency (DF) for each term
    for tokens_counter in documents.values():
        for term in tokens_counter:
            term_document_frequency[term] = term_document_frequency.get(term, 0) + 1
            
    # Calculating the inverted document frequency (IDF) for each term
    term_idf = {term: math.log(total_documents / df) for term, df in term_document_frequency.items()}
    return term_idf


def build_index(documents, term_weights, output_dir):
    index_inverted = {}                                         # Dictionary to store the inverted indices
    if not os.path.exists(output_dir):                          # Check if output directory exists
        os.makedirs(output_dir)                                 # If not, create the directory
    postings_file_path = os.path.join(output_dir, "postings.txt")  # Path for postings.txt
    with open(postings_file_path, 'w') as file:
        for doc_id, tokens_counter in documents.items():        # Iterating through each document and its tokens
            max_tf = max(tokens_counter.values())               # Maximum term frequency in the document
            for term, tf in tokens_counter.items():
                if term in term_weights:                        # Checking if the term has a term-weight
                    tfidf = (0.5 + 0.5 * (tf / max_tf)) * term_weights[term]  # Normalized TF-IDF value (using BM25 formula)
                    if term not in index_inverted:
                        index_inverted[term] = []               # Initializing the postings list if the term is absent
                    index_inverted[term].append((doc_id, tfidf))  # Adding the doc id and normalized tf-idf value to the postings list
                    file.write(f"{doc_id} , {tfidf:.2f}\n")     # Write to postings.txt
    
    # Sorting the postings by document ID and then by the tf-idf scores
    for postings_list in index_inverted.values():
        postings_list.sort(key=lambda x: (int(x[0]), x[1]))
        
    return index_inverted



def write_to_dictionary_file(output_dir, inverted_index):
    if not os.path.exists(output_dir):                                   # Checking if output path exists before writing to the output files
        os.makedirs(output_dir)
    dictionary_filepath = os.path.join(output_dir, 'dictionary.txt')     # Constructing the file path for the dictionary.txt output file
    with open(dictionary_filepath, 'w', encoding='utf-8') as dict_file:
        current_pos = 0                                                  # The start position of postings in the postings.txt file
        for term, postings in sorted(inverted_index.items()):
            num_docs = len(postings)
            dict_file.write(f"{term}\n{num_docs}\n{current_pos}\n")      # Writing the inverted index to the dictionary.txt file
            current_pos += num_docs


def index(input_dir, output_dir):
    start_time = time.time()                                             # Start time of processing the documents
    documents = processing_documents(input_dir)                          # Processing the documents from the input directory
    term_weights = calculate_term_weights(documents)                     # Calculating the term weights
    inverted_index = build_index(documents, term_weights, output_dir)    # Building and writing the inverted index to the dictionary file
    write_to_dictionary_file(output_dir, inverted_index)
    end_time = time.time()                                               # End time of processing the documents
    total_time = end_time - start_time                                   # Calculating the total time for the entire process
    return total_time


def plot_elapsed_time(num_documents_processed, elapsed_times):
    plt.plot(num_documents_processed, elapsed_times, marker='o')         # Plotting the elapsed times against the number of documents processed
    plt.title('Elapsed Time (in seconds) vs. Number of Documents Processed')
    plt.xlabel('Number of Documents Processed')
    plt.ylabel('Elapsed Time (in seconds)')
    plt.grid(True)                                                       # to display the grid lines
    plt.show()                                                           # To plot the graph


def main():
    parser = argparse.ArgumentParser(description="Build an inverted file index")                 # Creating an argument parser object
    parser.add_argument("input_dir", help="Input directory containing .txt files")
    parser.add_argument("output_dir", help="Output directory to store the dictionary and postings files")
    args = parser.parse_args()                                                                   # Parsing the command line arguments
    elapsed_times = []                      # To track the elapsed time for each set of documents processed
    num_documents_processed = []            # To track the number of documents processed
    
    # Processing documents in sets of 10, 20, 40, 80, 100, 200, 300, 400, 500
    for num_of_docs in [10, 20, 40, 80, 100, 200, 300, 400, 500]:
        elapsed_time = index(args.input_dir, args.output_dir)
        num_documents_processed.append(num_of_docs)
        elapsed_times.append(elapsed_time)
        print(f"Time taken to process {num_of_docs} documents: {elapsed_time:.2f} seconds")
    
    # Plotting the graph of the elapsed time versus the number of documents processed
    plot_elapsed_time(num_documents_processed, elapsed_times)
    # Calculating the total time to process all 503 documents
    total_elapsed_time = sum(elapsed_times)
    print(f"Total time required to process all 503 documents: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()