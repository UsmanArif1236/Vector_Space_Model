import os
import math
import json
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import streamlit as st

# Stop Words From file
stop_words_list = [
    'a', 'is', 'the', 'of', 'all', 'and', 'to', 'can', 'be', 'as', 'once', 'for', 'at', 'am', 'are', 'has', 'have', 'had', 'up', 'his', 'her', 'in', 'on', 'no', 'we', 'do'
]

porter_stemmer = PorterStemmer()

#This function tokenizes the text and removes stop words from it and perform stemming using porter stemmer.
def Pre_processing(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Filtering tokens by removing URLS,numbers,stopwords and converting to lowercase.
    filtered_tokens = []
    for token in tokens:
        if not token.startswith("http://") and not token.startswith("https://"):
            if not any(char.isdigit() for char in token):
                lower_token = token.lower()
                if lower_token not in stop_words_list:
                    filtered_tokens.append(lower_token)

    # Performed stemming on Filtered tokens.
    stemmed_tokens = []
    for token in filtered_tokens:
        stemmed_token = porter_stemmer.stem(token)
        stemmed_tokens.append(stemmed_token)

    return stemmed_tokens

#Creation of Inverted Index by iterating over each file in the directory, reads its content, and preprocesses the text to extract terms,The terms extracted are added to the inverted index along with their corresponding document IDs and positions.
def Inverted_index(directory):
    inverted_index = {}
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r") as file:
            text = file.read()
            terms = Pre_processing(text)
            doc_id = filename
            for position, term in enumerate(terms):
                if term not in inverted_index:
                    inverted_index[term] = {}
                if doc_id not in inverted_index[term]:
                    inverted_index[term][doc_id] = []
                inverted_index[term][doc_id].append(position)
    return inverted_index

# This function saves the indexes to a JSON file
def Save_index_to_JSON(inverted_index, filename):
    with open(filename, 'w') as file:
        json.dump(inverted_index, file)

# This function loads the indexes from a JSON file
def Load_index_from_JSON(filename):
    with open(filename, 'r') as file:
        index_file = json.load(file)
    return index_file

# TF-IDF Calculation
def Calculate_tf_idf(inverted_index, total_documents):
    tf_idf_weights = {}
    for term, postings in inverted_index.items():
        document_frequency = len(postings)
        idf = math.log10(total_documents / document_frequency)
        tf_idf_weights[term] = {}
        for doc_id, positions in postings.items():
            tf = len(positions)
            tf_idf = tf * idf
            tf_idf_weights[term][doc_id] = tf_idf
    return tf_idf_weights

# This function computes the vector representation of each document based on TF-IDF weights and the inverted index.
def Document_vector_representation(inverted_index, tf_idf_weights, directory):
    document_vectors = {}
    for filename in os.listdir(directory):
        doc_id = filename
        document_vectors[doc_id] = []
        for term in inverted_index.keys():
            tf_idf = tf_idf_weights.get(term, {}).get(doc_id, 0)
            document_vectors[doc_id].append(tf_idf)
    return document_vectors

# This function computes the vector representation of the query based on TF-IDF weights and the inverted index.
def Query_vector_representation(preprocessed_query, inverted_index, tf_idf_weights):
    query_vector = []
    for term in inverted_index.keys():
        if term in preprocessed_query:
            idf = math.log10(len(tf_idf_weights) / len(inverted_index[term]))
            tf_idf = 1 * idf
            query_vector.append(tf_idf)
        else:
            query_vector.append(0)
    return query_vector

# Cosine Similarity Calculation
def Cosine_similarity(query_vector, document_vectors):
    similarity_scores = {}

    for doc_id, document_vector in document_vectors.items():
        # Dot product of query and document vectors
        dot_product = 0
        for q, d in zip(query_vector, document_vector):
            dot_product += q * d
        
        # Magnitudes of the query and document vectors
        query_magnitude = 0
        for q in query_vector:
            query_magnitude += q ** 2
        query_magnitude = math.sqrt(query_magnitude)

        document_magnitude = 0
        for d in document_vector:
            document_magnitude += d ** 2
        document_magnitude = math.sqrt(document_magnitude)
        
        if query_magnitude != 0 and document_magnitude != 0:
            similarity = dot_product / (query_magnitude * document_magnitude)
        else:
            similarity = 0

        similarity_scores[doc_id] = similarity

    return similarity_scores

# This function ranks documents based on similarity scores and a given threshold.
def Rank_documents(similarity_scores, threshold):
    sorted_documents = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    top_documents = []
    for doc_id, score in sorted_documents:
        if score > threshold:
            top_documents.append(doc_id)
    return top_documents

# Main function
def main():
    global inverted_index_dict, tf_idf_weights_dict, document_vectors_dict
    directory = r"C:\Users\uarif\OneDrive\Documents\Semester 6\Information Retrieval\Assignment 01\ResearchPapers\ResearchPapers"
    
    total_documents = len(os.listdir(directory))
    print(total_documents)
    if os.path.exists("inverted_index.json") and os.path.exists("tf_idf_weights.json") and os.path.exists("document_vectors.json"):
        inverted_index_dict = Load_index_from_JSON("inverted_index.json")
        tf_idf_weights_dict = Load_index_from_JSON("tf_idf_weights.json")
        document_vectors_dict = Load_index_from_JSON("document_vectors.json")
    else:
        inverted_index_dict = Inverted_index(directory)
        total_documents = len(os.listdir(directory))
        tf_idf_weights_dict = Calculate_tf_idf(inverted_index_dict, total_documents)
        document_vectors_dict = Document_vector_representation(inverted_index_dict, tf_idf_weights_dict, directory)
        
        Save_index_to_JSON(inverted_index_dict, "inverted_index.json")
        Save_index_to_JSON(tf_idf_weights_dict, "tf_idf_weights.json")
        Save_index_to_JSON(document_vectors_dict, "document_vectors.json")

    st.title("Vector Space Model by Usman Arif")

    query = st.text_input("Enter your query:")
    threshold = st.slider("Threshold value (0 to 1):", min_value=0.0, max_value=1.0, step=0.01)

    if st.button("Search"):
        preprocessed_query = Pre_processing(query)
        query_vector = Query_vector_representation(preprocessed_query, inverted_index_dict, tf_idf_weights_dict)
        similarity_scores = Cosine_similarity(query_vector, document_vectors_dict)
        top_documents = Rank_documents(similarity_scores, threshold)
        if top_documents:
            st.write("Top documents:")
            for doc in top_documents:
                st.write(doc)
        else:
            st.write("No documents found.")

if __name__ == "__main__":
    main()
