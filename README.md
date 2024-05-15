# Vector_Space_Model

This project implements a Vector Space Model (VSM) for information retrieval using TF-IDF (Term Frequency-Inverse Document Frequency) weighting scheme. It allows users to enter a query, and based on the TF-IDF scores of terms in documents, retrieves relevant documents ranked by cosine similarity.


## Features
Pre-processing of text: Tokenization, stop words removal, and stemming using NLTK.
Creation of an inverted index to efficiently store and retrieve terms, document IDs, and positions.
Calculation of TF-IDF weights for terms in documents.
Representation of documents and queries as vectors based on TF-IDF weights.
Calculation of cosine similarity between query and documents to rank them.
Graphical User Interface (GUI) implemented using Streamlit for easy interaction.

## usage
1. Clone the repository to your local machine.
2. Run the Vector_Space_Model.py script using the following command:
    streamlit run Vector_Space_Model.py
3. This will launch the Streamlit web application in your default web browser.
4. Enter your query in the provided text input field.
5. Adjust the threshold value using the slider (range: 0 to 1).
6. Click on the "Search" button to retrieve relevant documents.
7. The top documents matching your query will be displayed.