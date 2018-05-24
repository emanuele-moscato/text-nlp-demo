import pdftotext
from os import listdir
from os.path import join, isfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

DATA_DIR = '../data/'
OUTPUT_DIR = '../output/'

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        top_words = " ".join(
            [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        )
        print("Topic #{}: {}".format(topic_idx, top_words))
    # print()

def main():
    input_files = [
        filename for filename in listdir(DATA_DIR)
            if isfile(join(DATA_DIR, filename))
    ]
    input_pdfs = [
        filename for filename in input_files
            if filename.split('.')[-1]=='pdf'
    ]
    
    documents = {}
    
    for filename in input_pdfs:
        with open(join(DATA_DIR, filename), 'rb') as f:
            pdf = pdftotext.PDF(f)
        
        #documents[filename] = ''.join(pdf)
        documents[filename] = pdf
        
    lda_topics = {}
    
    for key in documents.keys():
        tf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_df=0.95,
            min_df=2,
            use_idf=False,
            max_features=5000
        )
        tf_vectors = tf_vectorizer.fit_transform(documents[key])
        lda = LatentDirichletAllocation(
            n_components=5,
            max_iter=20,
            random_state=42,
            learning_method='batch'
        )
        lda_vectors = lda.fit_transform(tf_vectors)
        
        topics_topwords = []
        n_top_words = 3
        feature_names = tf_vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(lda.components_):
            top_words = " ".join(
                [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            )
            topics_topwords.append(top_words)
        lda_topics[key] = topics_topwords[0]
        
        print('Document: '+key)
        print_top_words(lda, tf_vectorizer.get_feature_names(), 3)
        print('Main topic: '+lda_topics[key]+'\n')
        

if __name__=="__main__":
    main()