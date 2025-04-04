=
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualization
import nltk # natural language processing
import string # string operations
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir('/kaggle/input/sms-spam-collection-dataset'))
=
import pandas as pd
df = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding='ISO-8859-1')
df.head()

df.columns

df.shape
df.info

#drop last 3 columns
df=df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])

df

df.sample(5)

#rename columns
df.rename(columns={'v1':'target','v2':'text'},inplace=True)

df.head(5)

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

df['target'].value_counts()

encoder.fit_transform(df['target'])

df['target']=encoder.fit_transform(df['target'])

df.head()

#missing values
df.isna().sum()

df.duplicated().sum()

# remove duplicate values
df.drop_duplicates(keep='first')
#Keeps the first occurrence of each duplicate row.

df=df.drop_duplicates(keep='first')

df.duplicated().sum()

df.shape

df.head()

df['target'].value_counts()

import matplotlib.pyplot as plt

plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")

#data is imbalanced
#3 now we will find in our dataset there are how many alphabets,words,sentences

import nltk

nltk.download('punkt')

df['text'].apply(len)#no.of characters

df['num_characters']=df['text'].apply(len)

df.head()

#no. of words
df['text'].apply(lambda x:nltk.word_tokenize(x))

# i want its length
df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))

df.head()

df['text'].apply(lambda x:nltk.sent_tokenize(x))

df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

df.head()

df[['num_characters', 'num_words', 'num_sentences']].describe()

df[df['target'] == 0][['num_characters', 'num_words', 'num_sentences']].describe()

# ham

df[df['target'] == 1][['num_characters', 'num_words', 'num_sentences']].describe()
#spam

import seaborn as sns

plt.figure(figsize=(12,8))
sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'],color='red')

# ham messages are made from more characters

plt.figure(figsize=(12,8))
sns.histplot(df[df['target']==0]['num_words'])
sns.histplot(df[df['target']==1]['num_words'],color='red')

sns.pairplot(df,hue='target')

df.columns

df_new = df[['target', 'num_characters', 'num_words', 'num_sentences']]

df_new

df_new.corr()

sns.heatmap(df_new.corr(),annot=True)

#num_characters	num_word num_sentences inka apas mai bohot acha correlation hai
#isliye sabko nahi rakh sakte
#isilye num_characters ko lenge kyuki uska corr sabse jyada hai target ke sath

"""Data preprocess
* lower case
* tokenization
* remove special characters
* remove stop words and punctuation
* stemming
"""

def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize words

    y = []
    for i in text:
        if i.isalnum():  # Fix: Add parentheses to `isalnum()`
            y.append(i)

    text = y[:]  # Copy filtered tokens
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]  # Copy filtered tokens
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)  # Join words into a cleaned sentence

from nltk.corpus import stopwords
stopwords.words('english')[:6]

import string
string.punctuation

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
ps.stem('dancing')

transform_text('Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...')

df['text'][0]

df['transformed_text']=df['text'].apply(transform_text)

df.head()

#wordcloud
from wordcloud import WordCloud
wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')

span_wc=wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))

plt.imshow(span_wc)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.imshow(span_wc)

ham_wc=wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.imshow(ham_wc)

df.head()

spam_corpus=[]
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

len(spam_corpus)

from collections import Counter
Counter(spam_corpus).most_common(30)

pd.DataFrame(Counter(spam_corpus).most_common(30))

sns.barplot(x=(pd.DataFrame(Counter(spam_corpus).most_common(30)))[0],y=(pd.DataFrame(Counter(spam_corpus).most_common(30)))[1])
plt.xticks(rotation='vertical')
plt.show()

ham_corpus=[]
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

len(ham_corpus)

from collections import Counter

pd.DataFrame(Counter(ham_corpus).most_common(30))

sns.barplot(x=(pd.DataFrame(Counter(ham_corpus).most_common(30)))[0],y=(pd.DataFrame(Counter(ham_corpus).most_common(30)))[1])
plt.xticks(rotation='vertical')
plt.show()

"""model building"""

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x=cv.fit_transform(df['transformed_text']).toarray()

x.shape

y=df['target'].values

y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()

gnb.fit(x_train,y_train)
y_pred1=gnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))

mnb.fit(x_train,y_train)
y_pred2=mnb.predict(x_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))

bnb.fit(x_train,y_train)
y_pred3=bnb.predict(x_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()

x=tfidf.fit_transform(df['transformed_text']).toarray()

y=df['target'].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()

gnb.fit(x_train,y_train)
y_pred1=gnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))

mnb.fit(x_train,y_train)
y_pred2=mnb.predict(x_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))

bnb.fit(x_train,y_train)
y_pred3=bnb.predict(x_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))

"""#tfidf-->mnb"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)

clfs = {
    'SVC' : svc,
    'KN' : knc,
    'NB': mnb,
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'AdaBoost': abc,
    'BgC': bc,
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}

def train_classifier(clf,x_train,y_train,x_test,y_test):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)

    return accuracy,precision

train_classifier(svc,x_train,y_train,x_test,y_test)

accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():

    current_accuracy,current_precision = train_classifier(clf, x_train,y_train,x_test,y_test)

    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)

    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)

performance_df

performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")

performance_df1

sns.catplot(x = 'Algorithm', y='value',
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()

# model improve
# 1. Change the max_features parameter of TfIdf

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)

new_df = performance_df.merge(temp_df,on='Algorithm')

new_df_scaled = new_df.merge(temp_df,on='Algorithm')

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)

new_df_scaled.merge(temp_df,on='Algorithm')

# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')

voting.fit(x_train,y_train)