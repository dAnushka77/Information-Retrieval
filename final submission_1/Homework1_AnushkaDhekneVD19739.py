# Importing all the necessary libraries

#nltk.download('punkt')                 # Installation for using the nltk library resources 
import os                               # For file operations
import argparse                         # For parsing command-line arguments
from collections import Counter         # For calculating token and frequency counts
import re                               # For using regular expressions
import time                             # For calculating elapsed and CPU times
import matplotlib.pyplot as plt         # For plotting the graphs
import nltk                             # For tokenizing
from nltk.tokenize import word_tokenize # For tokenizing text into words
from bs4 import BeautifulSoup           # For tokenizing HTML documents

plt.rcParams['font.family'] = 'sans-serif'  # Setting the font family for the documents


def parsing_html_files(file_path):                  # Function to parse each HTML file and returning it into a parsed textual format
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:      # Opening the file in the read 'r' format
        soup = BeautifulSoup(file, 'html.parser')       # Getting the text from the HTML file and storing it into a variable 'soup' and parsing the file contents using the BeautifulSoup library
        text = soup.get_text()                          # Getting the file contents from variable 'soup' as stored in the above line of code
        text = re.sub(r"\d+", '', text)                 # Removing digits from the parsed text
        text = re.sub(r"'", '', text)                   # Removing apostrophes
        text = re.sub(r"_", '', text)                   # Removing underscores
        text = re.sub(r"-", '', text)                   # Removing hyphens
        text = re.sub(r"\.", '', text)                  # Removing full stops and backslashes
        text = re.sub(r"[,.?!()\[\]]", '', text)        # Remove all additional symbols
        text = re.sub(r"[^\w\s]", '', text)             # Remove non-alphabetic characters except spaces
    return text


def text_conversion(text):                              # Tokeninzing the text and converting the tokens to lowercase
    tokens = word_tokenize(text)
    lowercase_tokens = [i.lower() for i in tokens]
    return lowercase_tokens


def process_document(file_path, output_dir):                # Function for processing a single document: parse, tokenize, lowercase, and write output.
    text = parsing_html_files(file_path)                    # Parsing each HTML file
    tokens = text_conversion(text)                          # Tokenizing each HTML file
    base_name = os.path.basename(file_path)                 # Storing the basename of output path for output file.
    output_filename = os.path.splitext(base_name)[0] + '.txt'          # Storing the output files in a .txt formmat
    output_filepath = os.path.join(output_dir, output_filename)        # Building the entire file path for output files
    with open(output_filepath, 'w', encoding='utf-8') as outfile:      # Opening the file path where the output files need to be stored and sriting to those files
        outfile.write(' '.join(tokens))                     # Writing the tokenized text to the output file.
    return tokens


def write_first_last_lines(file_path, n):                   # Functiong to store the first and last n (n =  50 in this assignment) lines to separate files.
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()                            # Reading the lines of the files

    first_n_lines = lines[:n]
    last_n_lines = lines[-n:]                               # Splitting the file lines

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    first_n_file = f'{base_name}_first_{n}.txt'             # Storing the basename of the path for first and last 50 lines file.
    last_n_file = f'{base_name}_last_{n}.txt'

    with open(first_n_file, 'w', encoding='utf-8') as file:
        file.writelines(first_n_lines)                      # Writing the first 50 lines into a seperate file.
    with open(last_n_file, 'w', encoding='utf-8') as file:
        file.writelines(last_n_lines)                       # Writing the last 50 lines into a seperate file.


def process_documents_and_results(input_dir, output_dir):   # Process all documents in the input directory and generate frequency files.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokens_all = []
    elapsed_times = []
    cpu_times = []
    processed = []

    file_names = os.listdir(input_dir)                              # Getting a list of filenames in the input directory
    num_files = len(file_names)
    for i, file_name in enumerate(file_names, 1):                   # Iterating over each input file
        file_path = os.path.join(input_dir, file_name)              # Building the entire file path for each single file
        if os.path.isfile(file_path):
            start_elapsed_time = time.time()                        # Record start elapsed time
            start_cpu_time = time.process_time()                    # Record start CPU time
            tokens = process_document(file_path, output_dir)        # Process documents after starting the elapsed time and cpu time
            tokens_all.extend(tokens)                               # Extending the list of all tokens
            end_elapsed_time = time.time()                          # Record end elapsed time
            end_cpu_time = time.process_time()                      # Record end CPU time
            elapsed_time = end_elapsed_time - start_elapsed_time    # Finding elapsed time after timer stops when all documents are processed
            cpu_time = end_cpu_time - start_cpu_time                # Finding cpu time after timer stops when all documents are processed
            elapsed_times.append(elapsed_time)                      # Appending the elapsed times to the list
            cpu_times.append(cpu_time)                              # Appending the CPU times to the list
            processed.append(i)                                     # Appending the count of processed documents

    token_frequency = Counter(tokens_all)                           # Calculating the frequency counts

    # Writing the two frequency files
    # File 1 - Sorting by token
    with open(os.path.join(output_dir, 'tokens_sorted_by_token.txt'), 'w', encoding='utf-8') as file:       # Iterating over token frequencies sorted by token
        for token, freq in sorted(token_frequency.items()): 
            file.write(f"{token}: {freq}\n")                                                                # Write token and its frequency to the file

    # File 2 - Sorting by frequency
    with open(os.path.join(output_dir, 'tokens_sorted_by_frequency.txt'), 'w', encoding='utf-8') as file:   # Iterating over token frequencies sorted by freuencies
        for token, freq in token_frequency.most_common():
            file.write(f"{token}: {freq}\n")                                                                # Write token and its frequency to the file

    # Write first and last 50 lines for each file
    write_first_last_lines('D:/SEM_4/Project/ISO/output_files/tokens_sorted_by_frequency.txt',50)
    write_first_last_lines('D:/SEM_4/Project/ISO/output_files/tokens_sorted_by_token.txt',50)

    # Plot for Elapsed Time
    plt.figure(figsize=(10, 6))
    plt.plot(processed, elapsed_times, label='Elapsed Time', color='blue')      # Number of documents processed vs elapsed time
    plt.xlabel('Number of Documents Processed')
    plt.ylabel('Time in seconds')
    plt.title('Elapsed Time vs Number of Documents Processed')
    plt.legend()
    plt.grid(True)
    plt.show()                                                                  # Displays the first plot

    # Plot for CPU Time
    plt.figure(figsize=(10, 6))                                                 # Creates a new figure for the CPU time vs Number of Documents processed plot
    plt.plot(processed, cpu_times, label='CPU Time', color='orange')            # Number of documents processed vs CPU time
    plt.xlabel('Number of Documents Processed')
    plt.ylabel('Time in seconds')
    plt.title('CPU Time vs Number of Documents Processed')
    plt.legend()
    plt.grid(True)
    plt.show()                                                                  # Displays the second plot

    # Calculating the total time
    total_time = sum(elapsed_times)                                             # Finding the sum of the total elapsed time
    total_cpu_time = sum(cpu_times)                                             # Finding the sum of the total cpu time
    print_elapsed = print(f"Total elapsed time: {total_time:.2f} seconds")      # Printing total times upto 2 decimal digits
    print_cpu = print(f"Total CPU time: {total_cpu_time:.2f} seconds")  
    
    return print_elapsed, print_cpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTML Document Tokenizer")                 # Creating an argument parser object
    # Adding command line arguments for the input and output directories
    parser.add_argument("input_dir", help="Directory containing all the input HTML files")
    parser.add_argument("output_dir", help="Directory where all the output files are stored")
    arguments = parser.parse_args()                                                         # Parsing the command line arguments
    time_df = process_documents_and_results(arguments.input_dir, arguments.output_dir)      # Calling the function with the input and output directories