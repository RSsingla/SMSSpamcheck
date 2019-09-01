#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import sklearn
import numpy
import pandas
import sys
import os


# # #1.LOAD THE DATASET
# 

# In[2]:


os.chdir("C:/Users/kc/Desktop/SMSspam")


# In[3]:


import pandas as pd
import numpy as np

# load the dataset of SMS messages
df = pd.read_table('SMSSPamCollection', header=None, encoding='utf-8')


# In[4]:


# print useful information about the dataset
print(df.info())
print(df.head())


# In[5]:


# check class distribution class 0 is worked as an idenntifier
classes = df[0]
print(classes.value_counts())


# # #2.DATA PROCESSING

# In[6]:


from sklearn.preprocessing import LabelEncoder
# convert class labels to binary values, 0 = ham and 1 = spam
encoder = LabelEncoder()
Y = encoder.fit_transform(classes)

#Y is target element or say predictor
print(Y[0:50])


# In[7]:


# store the SMS message data and class 1 represent the text message in dataset
text_messages = df[1]
print(text_messages[:50])


# Now we deal with the regular expression in this special characters like '@' etc 
# We can also do this with a python inbuilt library for for regular expression 
# for this particular project we do not this library 

# DEAL WITH REGULAR EXPRESSION FOR EACH EMAIL,URL etc. we have differen format of regular expression here 'r' represent regular expression
# r'^.+@[^\.].*\.[a-z]{2,}$' - expression for email format
# r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$' - expression for URL format
# r'£|\$' - expression for money symbols
# r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$' - expression for phone number in USA format(3-3-4)
# r'\d+(\.\d+)?' - regular expression for any random number

# In[8]:


# use regular expressions to replace email addresses, URLs, phone numbers, other numbers
# Replace email addresses with 'email'
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                 'emailaddress')


# In[9]:


# Replace URLs with 'webaddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                  'webaddress')


# In[10]:


# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
processed = processed.str.replace(r'£|\$', 'moneysymb')


# In[11]:


# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr')


# In[12]:


# Replace numbers with 'numbr'
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')


# In[13]:


# Remove punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ')


# In[14]:


# Replace whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')


# In[15]:


# Remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')


# In[16]:


# change words to lower case - Hello, HELLO, hello are all the same word
processed = processed.str.lower()


# In[17]:


print(processed)


# ![image.png](attachment:image.png)

# In[18]:


from nltk.corpus import stopwords

# remove stop words from text messages

stop_words = set(stopwords.words('english'))

processed = processed.apply(lambda x: ' '.join(
    term for term in x.split() if term not in stop_words))


# The idea of stemming is a sort of normalizing method. Many variations of words carry the same meaning, other than when tense is involved.
# 
# The reason why we stem is to shorten the lookup, and normalize sentences.
# 
# Consider:
# 
# I was taking a ride in the car.
# I was riding in the car.
# This sentence means the same thing. in the car is the same. I was is the same. the ing denotes a clear past-tense in both cases, so is it truly necessary to differentiate between ride and riding, in the case of just trying to figure out the meaning of what this past-tense activity was?
# 
# No, not really.
# 
# This is just one minor example, but imagine every word in the English language, every possible tense and affix you can put on a word. Having individual dictionary entries per version would be highly redundant and inefficient, especially since, once we convert to numbers, the "value" is going to be identical.
# 
# One of the most popular stemming algorithms is the Porter stemmer, which has been around since 1979.

# In[19]:


# Remove word stems using a Porter stemmer
ps = nltk.PorterStemmer()

processed = processed.apply(lambda x: ' '.join(
    ps.stem(term) for term in x.split()))


# Pick out the bag of word that are most frequently used

# In[20]:


import nltk
nltk.download('punkt')


# In[21]:


from nltk.tokenize import word_tokenize

# create bag-of-words
all_words = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)


# In[22]:


# print the total number of words and the 15 most common words
print('Number of words: {}'.format(len(all_words)))
print('Most common words: {}'.format(all_words.most_common(15)))


# In[23]:


# use the 2000 most common words as features
word_features = list(all_words.keys())[:2000]


# In[24]:


# The find_features function will determine which of the 1500 word features are contained in the review
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

# Lets see an example!
features = find_features(processed[0])
for key, value in features.items():
    if value == True:
        print(key)


# In[25]:


# Now lets do it for all the messages
messages = zip(processed, Y)

# define a seed for reproducibility
seed = 1
np.random.seed = seed
#np.random.shuffle(messages)

# call find_features function for each SMS message
featuresets = [(find_features(text), label) for (text, label) in messages]


# In[26]:


# we can split the featuresets into training and testing datasets using sklearn
from sklearn import model_selection

# split the data into training and testing datasets
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)


# In[27]:


print(len(training))
print(len(testing))


# In[28]:


print(training)
print(testing)


# # #Scikit-Learn Classifiers with NLTK
# 

# In[29]:


# We can use sklearn algorithms in NLTK
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

model = SklearnClassifier(SVC(kernel = 'linear'))

# train the model on the training data
model.train(training)

# and test on the testing dataset!
accuracy = nltk.classify.accuracy(model, testing)*100
print("SVC Accuracy: {}".format(accuracy))


# In[30]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Define models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
    print("{} Accuracy: {}".format(name, accuracy))


# In[31]:


# Ensemble methods - Voting classifier
from sklearn.ensemble import VotingClassifier

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1))
#nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Voting Classifier: Accuracy: {}".format(accuracy))

