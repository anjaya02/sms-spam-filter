import pandas as pd
import pickle
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus       import stopwords
from nltk.stem.porter  import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes           import MultinomialNB
from sklearn.model_selection       import train_test_split

# 0. Load & clean
df = pd.read_csv("spam.csv", encoding="latin1").drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'])
df.rename(columns={'v1':'target','v2':'text'}, inplace=True)
df.drop_duplicates(inplace=True)
df['target'] = df['target'].map({'ham':0,'spam':1})

# 1. Text preprocessing
ps = PorterStemmer()
stop = set(stopwords.words('english'))

def transform(text):
    tokens = wordpunct_tokenize(text.lower())
    kept   = [t for t in tokens if t.isalnum() and t not in stop]
    stems  = [ps.stem(t) for t in kept]
    return " ".join(stems)

df['transformed'] = df['text'].apply(transform)

# 2. Vectorize
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed'])
y = df['target']

# 3. Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
model = MultinomialNB()
model.fit(X_train, y_train)

# 4. Save
pickle.dump(tfidf,  open("vectorizer.pkl","wb"))
pickle.dump(model,  open("model.pkl","wb"))

print("✅ Training complete, pickles saved.")
