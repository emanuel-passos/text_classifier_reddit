import re
import praw
import config
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

## loading datas

class_matters = ['datascience', 'machinelearning', 'physics', 'astrology', 'conspiracy']

def load_data():
    
    api_reddit = praw.Reddit(
            client_id="aF-1sE_k4VYFJVeUBp-1ag",
            client_secret="	_Ocb-pgK51QmayquDxMyyBZ5qNB0wQ",
            password="Epicduel10",
            user_agent="webscrp-appe",
            username="Majestic_Second2063",
    )
    
    char_count = lambda post: len(re.sub('\W\d', '', post.selftext))
    
    mask = lambda post: char_count(post) >= 100
    
    data = []
    labels = []
    
    for i, assunto in enumerate(class_matters):
        
        subreddit_data = api_reddit.subreddit(assunto).new(limit = 1000)
        
        posts = [post.selftext for post in filter(mask, subreddit_data)]
        
        data.extend(posts)
        labels.extend([i] * len(posts))
        
        print(f"Number of topic posts r/{assunto}: {len(posts)} ",
              f"\nextracted post: {posts[0][:600]}...\n", "_" * 80 + '\n'
              )
        
    return data, labels

## split into training and test data

TEST_SIZE = .2
RANDOM_STATE = 0

def split_data():
    
    print(f"Split {100 * TEST_SIZE}% data for model training and evaluation...")
    
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=TEST_SIZE, random_state= RANDOM_STATE)

    print(f"{len(y_test)} test sample")
    
    return x_train, x_test, y_train, y_test

## Data Pre-Processing and Attribute Extraction

MIN_DOC_FREQ = 1
N_COMPONENTS = 1000
N_ITER = 30 

def preprocessing_pipeline():
    
    pattern = r'\W|\d|http.*\S+|www.*\S+'
    preprocessor = lambda text: re.sub(pattern, '', text)
    
    vectorizer = TfidfVectorizer(preprocessor= preprocessor, stop_words= 'english', min_df = MIN_DOC_FREQ)
    
    decomposition = TruncatedSVD(n_components = N_COMPONENTS, n_iter = N_ITER)
    
    Pipeline = [('tfidf', vectorizer), ('svd', decomposition)]
    
    return Pipeline
    
## Selection model

N_NEIGHBORS = 4
CV = 3

def create_model():
    
    model_1 = KNeighborsClassifier(n_neighbors= N_NEIGHBORS)
    model_2 = RandomForestClassifier(random_state=RANDOM_STATE)
    model_3 = LogisticRegressionCV(cv= CV, random_state= RANDOM_STATE)
    
    models = [("KNN", model_1), ("RandomForest", model_2), ("LogReg", model_3)]
    
    return models

def treina_avalia(models, pipeline, x_train, x_test, y_train, y_test):
    
    results = []
    
    for name, model in models:
        
        pipe = Pipeline(pipeline + [(name, model)])
        
        print(f'Taining model {name} with training datas...')
        pipe.fit(x_train, y_train)
        
        y_pred = pipe.predict(x_test)
        
        report = classification_report(y_test, y_pred)
        print("Report of classification\n", report)
        
        results.append([model, {'model': name, 'previsões': y_pred, 'report': report}])
        
    return results

if __name__ == "__main__":
    
    data, labels = load_data()
    
    x_train, x_test, y_train, y_test = split_data()
    
    pipeline = preprocessing_pipeline()
    
    all_models = create_model()
    
    results = treina_avalia(all_models, pipeline, x_train, x_test, y_train, y_test)

print("successfully concluded")

## visualização
    
def plot_distribution():
    
    _, counts = np.unique(labels, return_counts= True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize = (15, 6), dpi = 120)
    plt.title("Number of posts by subject")
    sns.barplot(x=assuntos, y=counts)
    plt.legend([' '.join([f.title(),f"-{c} posts"]) for f,c in zip(assuntos, counts)])
    plt.show()
    
def plot_confusion(result):
    print("Reports of classification\n", result[-1]['report'])
    y_pred = result[-1]['predications']
    conf_matrix = confusion_matrix(y_test, y_pred)
    _, test_counts =np.unique()
    conf_matrix_percent = conf_matrix / test_counts.transpose() * 100
    plt.figure(figsize = (9,8), dpi = 120)
    plt.title(result[-1]['modelo'].upper() + " Resultados")
    plt.xlabel("Valor Real")
    plt.ylabel("Previsão do Modelo")
    ticklabels = [f"r/{sub}" for sub in assuntos]
    sns.heatmap(data = conf_matrix_percent, xticklabels = ticklabels, yticklabels = ticklabels, annot = True, fmt = '.2f')
    plt.show()


# Gráfico de avaliação
plot_distribution()

# Resultado do KNN
plot_confusion(results[0])

# Resultado do RandomForest
plot_confusion(results[1])

# Resultado da Regressão Logística
plot_confusion(results[2])