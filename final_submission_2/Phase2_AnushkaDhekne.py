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

#Function to preprocess the text and remove stopwords
def preprocessing_text(text):
    # Tokenize the text by finding all words in the text and converting it to lower case
    tokens = re.findall(r'\b\w+\b', text.lower())
    # Removing stopwords, words with single occurrences, and 1 letter words
    tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    
    return tokens

#Function to calculate term weights for tokens in documents using TF-IDF and BM25 algorithms
def calculating_term_weights(input_dir, output_dir):  
    
    # Assigning variable constants for BM25
    b = 0.75
    k1 = 1.5

    total_start_time = time.time()                            # Starting the elapsed time
    
    # Variables to store document frequencies and lengths
    elapsed_times = []                                        # List to store elapsed times for each set of documents processed
    document_frequencies = Counter()                          # Counter to store document frequencies for each term
    document_lengths = {}                                     # Dictionary to store document lengths (num of tokens) for each document
    num_of_documents_processed = []                              # To track the actual number of documents processed
    token_frequencies_list = []                               # List to store token frequencies for BM25

    for file_name in os.listdir(input_dir):                   # Iterating over each file from the input directory
        file_path = os.path.join(input_dir, file_name)        # Constructing the full file path
        if os.path.isfile(file_path):                         # Check if path is a file
            with open(file_path, 'r', encoding='utf-8') as file:   
                fileread = file.read()                        # Read the contents of the file into the 'fileread' variable
            
            tokens = preprocessing_text(fileread)             # Preprocess the text from 'fileread'
            document_frequencies.update(set(tokens))          # Update document frequencies            
            document_lengths[file_name] = len(tokens)         # Update document lengths            
            token_frequencies_list.append(Counter(tokens))    # Store token frequencies for BM25 calculation

    # Finding the total number of documents
    num_of_documents = len(document_lengths)    
    # Finding the average document length                  
    average_document_length = sum(document_lengths.values()) / num_of_documents
    # Finding the inverse document frequencies (IDF values) for BM25
    idf_values = {token: math.log((num_of_documents - document_frequencies[token] + 0.5) / (document_frequencies[token] + 0.5)) for token in document_frequencies}

    # Iterating through each file of the input firectory
    for i, file_name in enumerate(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, file_name)               # Constructing the full file path
        if os.path.isfile(file_path):                                # Check if path is a file
            # Calculate the TF-IDF and BM25 scores for each token
            token_frequencies = token_frequencies_list[i]            # Get token frequencies for the current file
            tfidf_weights = {}                                       # Dictionary to store tf-idf scores for the tokens
            bm25_weights = {}                                        # Dictionary to store bm25 scores for the tokens
            
            for token, freq in token_frequencies.items():
                # TF-IDF
                tf = freq / len(tokens)                              # Finding the term frequency
                tfidf_weights[token] = tf * idf_values[token]        # Calculating and storing the tf-idf scores for each token 

                # BM25
                idf = idf_values[token]                              # Inverse document frequency (IDF) calculation
                bm25 = idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * document_lengths[file_name] / average_document_length))
                bm25_weights[token] = bm25                           # Storing the bm25 score for each token

            # Writing the token weights to the respective output files
            output_file_name = f"{os.path.splitext(file_name)[0]}.wts"
            output_file_path = os.path.join(output_dir, output_file_name)
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                # Writing the token weights for each token to the output file for both, tf-idf and bm25
                for token in token_frequencies:
                    outfile.write(f"{token}: TF-IDF - {tfidf_weights[token]} | BM25 - {bm25_weights[token]}\n")        
            i += 1                                                   # Incrementing the index counter for the num of docs processed

            # Checking if we have processed the given of documents (in the list below) to record elapsed times for each set of documents
            if i in [10, 20, 40, 80, 100, 200, 300, 400, 500]:
                # Calculating elapsed time for processing the current set of documents
                end_partial_time = time.time()
                elapsed_time_partial = end_partial_time - total_start_time
                # Recording the elapsed time and the number of documents processed
                elapsed_times.append(end_partial_time)
                num_of_documents_processed.append(i)

    # Recording the total elapsed time and the total number of documents processed
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    elapsed_times.append(total_elapsed_time)
    num_of_documents_processed.append(i)

    # Plotting the documents processed vs elapsed time graph
    plt.plot(num_of_documents_processed, elapsed_times, marker='o')
    plt.xlabel('Number of Documents Processed')
    plt.ylabel('Elapsed Time')
    plt.title('Elapsed Time Time vs. Number of Documents Processed')
    plt.grid(True)
    plt.show()
    
    # Printing the total elapsed time for all 503 documents upto 2 decimal digits
    print(f"Total time for processing all 503 documents: {total_elapsed_time:.2f} seconds")          


def main():
    parser = argparse.ArgumentParser(description="Calculating term weights for tokens in documents")    # Creating an argument parser object
    parser.add_argument("input_dir", help="Input Directory containing all .txt tokenized files")
    parser.add_argument("output_dir", help="Output directory to store token weights in .wts format")
    args = parser.parse_args()                                                                          # Parsing the command line arguments
    calculating_term_weights(args.input_dir, args.output_dir)                                           # Calling the function with the input and output directories

if __name__ == "__main__":                                                                              # Calling the main function
    main()
 