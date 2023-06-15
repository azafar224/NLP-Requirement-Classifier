**NLP Requirement Classifier**

The NLP Requirement Classifier is a Python script that uses Natural Language Processing (NLP) techniques and machine learning algorithms to classify requirements into different types. It applies various NLP techniques such as tokenization, stopword removal, lemmatization, named entity recognition, part-of-speech tagging, chunking, and dependency parsing to extract features from the requirements. The extracted features are then used to train a random forest classifier, which predicts the types of new requirements.

**Features**
        Preprocesses requirement text by tokenizing, removing stopwords, and lemmatizing.
        Extracts requirements and their types based on predefined prefixes.
        Performs named entity recognition (NER) to identify entities in requirement sentences.
        Assigns part-of-speech (POS) tags to each token in requirement sentences.
        Applies chunking to identify noun phrases in requirement sentences.
        Performs dependency parsing to determine syntactic relationships between words.
        Computes word frequency in requirement sentences.
        Utilizes TF-IDF vectorization for feature extraction.
        Trains a random forest classifier on the feature vectors.
        Evaluates classifier performance using cross-validation.
        Predicts requirement types for new requirements.
        Calculates accuracy score for the classifier.

**Installation**
        
   Ensure you have Python 3.x installed on your machine.
        
        Clone the repository:

        git clone https://github.com/azafar224/nlp-requirement-classifier.git
        
   Install the required dependencies:
       
       pip install nltk spacy scikit-learn
        
   Download the necessary resources for NLTK and spaCy:

        python -m nltk.downloader stopwords
        python -m nltk.downloader punkt
        python -m nltk.downloader averaged_perceptron_tagger
        python -m spacy download en_core_web_sm


**Usage**

   Place your requirement dataset files in the same directory as the code file. Each file should contain one requirement per line.

   Modify the train_dataset_file_path and test_dataset_file_path variables in the main() function of the code.py file to point to your train and test dataset files,            respectively.
   
   Run the script:

        python code.py
        
   The script will preprocess the requirements, apply NLP techniques, train the classifier on the training dataset, evaluate its performance using cross-validation, and        predict the requirement types for the test dataset. The predicted types will be printed along with the original requirements, and the accuracy score of the classifier      will be displayed.

   Please ensure that your dataset files adhere to the required format, where each requirement is prefixed with its type (e.g., "F:", "A:", "FT:", etc.), for accurate          results.        
   
   
**Contributing**

   Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

**Acknowledgements**

   The NLP Requirement Classifier was developed using the following libraries and resources:

        NLTK (Natural Language Toolkit): https://www.nltk.org/
        spaCy: https://spacy.io/
        scikit-learn: https://scikit-learn.org/

   The project was inspired by the need for automating requirement classification tasks and leveraging NLP techniques to improve efficiency and accuracy.

**Contact**
For any questions or inquiries, please contact [ahmadzafar224@gmail.com].

Thank you for using the NLP Requirement Classifier! We hope it helps streamline your requirement analysis process and improves your project outcomes.
     
