import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Importation de données
movie_data = load_files(r"C:\Users\33763\Desktop\Classif - projet FoDo\documentsClassificator\txt_sentoken")
X, y = movie_data.data, movie_data.target

documents = []
stemmer = WordNetLemmatizer()

# Nettoyage de données
for sen in range(0, len(X)):
     # Supprimer tous les caractères spéciaux
     document = re.sub(r'\W', ' ', str(X[sen]))
    
     # Supprimer tous les caractères uniques
     document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
     # Supprimer les caractères uniques du début
     document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
     # Remplacement de plusieurs espaces par un seul espace
     document = re.sub(r'\s+', ' ', document, flags=re.I)
    
     # Suppression du préfixe «b»
     document = re.sub(r'^b\s+', '', document)
    
     # Conversion en minuscules
     document = document.lower()
    
     # Lemmatisation
     document = document.split()

     document = [stemmer.lemmatize(word) for word in document]
     document = ' '.join(document)
    
     documents.append(document)

# Conversion de text en nombre
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

# Calculer TF-IDF
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

# Ensemble d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Algorithme de classification - Random Forest
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test)

# Evaluation du modèle 
print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

