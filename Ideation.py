# -*- coding: utf-8 -*-
"""
# **IMPORTING NECESSARY LIBRARIES**
"""

# pip install unidecode

# pip install contractions

# Data processing
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Text Processing
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import unidecode
import contractions

# Deep Learning
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, InputLayer, Flatten, GRU
from keras.models import Sequential
from keras.optimizers import Adam

import unidecode
import contractions

# Downloading NLTK dependencies
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

"""# **Loading Dataset (from google drive path)**"""

# Loading dataset, selecting 2nd and 3rd columns only
data = pd.read_csv('/content/drive/MyDrive/Suicide_Detection.csv',usecols=[1,2],encoding='utf-8')

# Creating backup of original dataframe
data_backup = data.copy()

# Viewing subset of original data
data.groupby("class").sample(n=5, random_state=99)

"""# **Exploratory Data Analysis**"""

# Checking dataset shape
data_shape = data.shape
print(f'The dataset has {data_shape[0]} rows and {data_shape[1]} columns')

# Checking for missing values
data.isna().sum()

# Checking for duplicate values
duplicates = data[data.duplicated]
print(f'Dataset contains {len(duplicates)} duplicated value(s)')

# Visuaising count of dataset classes
count_ax = sns.countplot(data=data, x='class', hue='class', palette=sns.palettes.mpl_palette('Dark2'))
# Add count at top of each bar
count_ax.bar_label(count_ax.containers[0])
count_ax.bar_label(count_ax.containers[1])
# Removing top and right graph spines
plt.gca().spines[['top', 'right',]].set_visible(False)

def mean_word_count(frame, category, category_col='class', text_col='text'):
  """
  This function computes the mean number of words in the texts of each class
  params:
  frame: Dataframe. type: pd.DataFrame
  category: name of label to compute mean. type: str
  category_col: name of colun containing labels. type:str, default = 'class'
  text_col: name of column containing texts. type:str, default = 'text'

  return: List containing mean and list of word counts
  """
  # Query the dataframe to obtain count of specific column
  counts = frame[frame[category_col]==category][text_col].str.split().str.len()
  # Calculate mean of specific column
  category_mean = np.mean(counts)
  # List of word counts
  word_count = counts.to_list()

  return [category_mean, word_count]

# Obtain mean number of words for suicide texts
s_text = mean_word_count(frame=data, category='suicide')
print(f'The average number of words in a suicide text is {round(s_text[0],2)}')

# Obtain mean number of words for non-suicide texts
ns_text = mean_word_count(frame=data, category='non-suicide')
print(f'The average number of words in a non-suicide text is {round(ns_text[0],2)}')

# Obtaining list of text lengths
s_word_count = s_text[1]
ns_word_count = ns_text[1]

# PLOTTING WORD-COUNT histogram of both classes
fig,ax = plt.subplots(1,2,figsize=(10,4), sharey=True)
sns.histplot(s_word_count,color='red', bins=50, ax=ax[0]).set(title='Suicide Text')
sns.histplot(ns_word_count,color='green', bins=50, ax=ax[1]).set(title='Non-Suicide Text')

# Set plot limitand labels
plt.setp(ax, xlim=(0,1750), xlabel= 'No. of Words', ylabel='Count')
# set plot main title
fig.suptitle('Words per text')

print(f'Suicide class:\n Longest text: {max(s_word_count)}\n Shortest Text: {min(s_word_count)}')
print()
print(f'Non-Suicide class:\n Longest text: {max(ns_word_count)}\n Shortest Text: {min(ns_word_count)}')

"""# **Data Cleaning and Preprocessing**

## Data Cleaning
"""

# Computing texts with strings
links= data_backup[data_backup['text'].str.contains('https')]

# Viewing texts with links by class
sns.countplot(data=links, x='class', hue='class', palette=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

# Converting all text to lowercase
data['clean_text'] = data['text'].str.lower()

# Removing trailing whitespaces in text
data['clean_text'] = data['clean_text'].str.strip()

def text_cleaner(text):
  """
  This function performs multiple text preprocessing techniques
  """
  # Remove links
  link_regex = r'https?\S+|(www)?\.?\w+\.com'
  text = re.sub(pattern=link_regex, repl=' ', string=text)

  # Remove Emojis
  text = unidecode.unidecode(text)

  # Expand contractions
  text = contractions.fix(text)

  # remove dates
  date_regex = r'\d+\/\d+\/\d+'
  text = re.sub(pattern=date_regex, repl=' ', string=text)

  # Remove punctuations
  punctuation_map = str.maketrans('', '', string.punctuation)
  text = text.translate(punctuation_map)

  # Remove Specific words
  words_regex = r'trigger|warning|filler|disclaimer|interactive quiz|link|suicide'
  text = re.sub(pattern=words_regex, repl=' ', string=text)

  # Remove  numbers
  digit_regex = r'\d+'
  text = re.sub(pattern=digit_regex, repl=' ', string=text)

  # Remove Single letters
  letter_regex = r'\s+[a-z]\s+'
  text = re.sub(pattern=letter_regex, repl=' ', string=text)

  # Remove extra spaces
  space_regex = r'\s+'
  text = re.sub(pattern=space_regex, repl=' ', string=text)

  # Strip texts
  text = text.strip()

  return text

# Apply data cleaning function
data['clean_text'] = data['clean_text'].apply(text_cleaner) # 2minutes 32 seconds

# viewing subset of dataset after cleaning
data.groupby("class").sample(n=5, random_state=99)

# After cleaning, checking for entries with five words or less
five_or_less_entries = data[data['clean_text'].str.split().str.len() <= 5]
five_or_less_indexes = list(five_or_less_entries.index)

# Class distribution of <5 entries
five_or_less_entries['class'].value_counts()

# Dropping entrie with five or less entires
data.drop(index=five_or_less_indexes, inplace=True)

# Checking data balance after dropping rows
data['class'].value_counts(normalize=True)*100

print(f'Final number of rows for model: {data.shape[0]}')

"""## Data Preprocessing"""

# Encoding Classes
data['class'] = data['class'].apply(lambda entry: 0 if entry == 'non-suicide' else 1)

# Defining stopword list
stop_list = stopwords.words('english')

# Function for removing stop words
def stopword_remover(text, stopword_list):
  """
  This function removes all stopwords from text
  params:
  text: text to reomve stopwords. type:str
  stopword_list: list of stopwords for comparison

  return new_text: text without stopwords
  """
  # empty list to store non stopwords
  no_stopwords = []
  # Tokenize text and Loop through all words in text
  for word in word_tokenize(text):
    # keep non stopwords
    if word not in stopword_list:
      no_stopwords.append(word)

  # convert back to string from list
  new_text = ' '.join(no_stopwords)
  return new_text

# Removing stop words from data text
data['clean_text'] = data['clean_text'].apply(lambda tokens: stopword_remover(text=tokens, stopword_list=stop_list))

# Function to Lemmatize text
def text_lemma(text):
  """
  This function lemmatizes words in text
  """
   # Initialize the lemmatizer
  lemma = WordNetLemmatizer()
  # List to store lemmtized version of words
  lemmatized_words = []

  # loop throuh words in text and lemmatize
  for word in text.split():
    lemmatized_words.append(lemma.lemmatize(word, pos='v'))

  # # convert list of words back to string
  lemmatized_text = ' '.join(lemmatized_words)
  return lemmatized_text

# Apply Lemmatization Function
data['clean_text'] = data['clean_text'].apply(lambda text: text_lemma(text))

# Word cloud
def genWordCloud(hashtags):
  """
  This function creates a word cloud
  params:
  hashtags: list of words or sereis
  """
    # Read the whole text.
  text = ' '.join(hashtags)

  # Generate a word cloud image
  wordcloud = WordCloud(stopwords=stop_list, width =520,
                        height =260, max_font_size=40).generate(text)

  # Display the generated image:
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")

  '''plt.figure()
  plt.imshow(wordcloud, interpolation="bilinear")
  plt.axis("off")
  plt.savefig('wordcloud_all.pdf', dpi=500)
  plt.show()'''

# Create column for tokenized texts
data['tokenized'] = data['clean_text'].apply(lambda x: word_tokenize(x))

# Creating word cloud for non-suicide texts
nsui = data[data['class']==0]
genWordCloud(nsui['clean_text'])

# Creating word cloud for suicide texts
sui = data[data['class']==1]
genWordCloud(sui['clean_text'])

"""# **Building Models**"""

def model_eval(actual_vals, pred_vals):
  """
  This function generates a classification report and
  plots the confusion matrix of a classifier.
  params:
  actual_vals: true values of prediction. type: list
  pred_vals: predicted values. type: list
  """
  # Generate classification report
  eval_report = classification_report(y_true= actual_vals, y_pred= pred_vals)
  # Generate confusion matrix
  model_conf = confusion_matrix(y_true=actual_vals, y_pred=pred_vals)

  # Output classification report
  print(eval_report)
  # Display heatmap
  sns.heatmap(model_conf, annot=True ,fmt='g', xticklabels=['Non-Suicide', 'Suicide'], yticklabels=['Non-Suicide', 'Suicide'])

"""## Traditional models with Tf-idf vectorizations"""

# Split into training and test data for Traditional models
train_texts, test_texts, train_cat, test_cat = train_test_split(data['clean_text'], data['class'], test_size=0.4, random_state=16, stratify=data['class'])

# Initizalize the vectorizer
tf_idf_vectorizer = TfidfVectorizer(use_idf=True, max_features=300, ngram_range=(1,3), max_df=0.4)

# Fit and transform the training data with the vectorizer
train_texts_tfidf = tf_idf_vectorizer.fit_transform(train_texts)
# Transform the testing data
test_texts_tfidf = tf_idf_vectorizer.transform(test_texts)

"""### Logistic Regression model"""

# Initialize Logistic regrssion model
logistic_model = LogisticRegression(max_iter=1000, random_state=61)
# Fitting the logistic model to training data
logistic_model.fit(train_texts_tfidf, train_cat)

# using logistic model to predict
logistic_predictions = logistic_model.predict(test_texts_tfidf)

# Evaluating model
model_eval(actual_vals=test_cat, pred_vals=logistic_predictions)
plt.title('Logistic Predictions')

# Exploring the top vectors words used by model
top_words = np.argsort(logistic_model.coef_)[:, -50:]

# create tfidf vocabulary dictionary
tf_voc = tf_idf_vectorizer.vocabulary_
# Reverse Dictionary order
inv_voc = {v: k for k, v in tf_voc.items()}

# Obtain corresponding words for model vector
log_words = [inv_voc[x] for x in top_words[0]]

# Create word cloud of Logistic model top words
genWordCloud(log_words)

"""### Naive Bayes model"""

# Define parameter of possile alpha values
naive_param={'alpha': [0.2, 0.4, 0.6, 0.8, 1],
             'fit_prior':[True, False]}
# Using gridsearch to obtain best alpha value
naive_gridsearch = GridSearchCV(estimator=MultinomialNB(), param_grid=naive_param)

# Fitting gridsearchcv to trian data
naive_gridsearch.fit(train_texts_tfidf, train_cat)

print(f'The best score for the Naive Bayes:{naive_gridsearch.best_score_} is achived with parameters: {naive_gridsearch.best_estimator_}')

# Training the Naive Bayes Model with best hperparameters
naive_model = MultinomialNB(alpha=0.6, fit_prior=False)
# Fitting the model
naive_model.fit(train_texts_tfidf, train_cat)

# Predicting with naive bayes model
naive_predictions = naive_model.predict(test_texts_tfidf)

# Evaluating Naive Bayes model
model_eval(actual_vals=test_cat, pred_vals=naive_predictions)
plt.title('Naive Bayes Predictions')

"""## **Building Deep Learning models**

### Creating Helper Functions
"""

def eval_deep_model(history, loss=False, acc=False):
  """
    This function plots the accuray and loss comparison for training and
    validation data of deep learning models
    params:
    history: model history
  """
  # Convert history to dataframe
  model_df = pd.DataFrame(history.history)
  epochs = len(model_df)
  lim = list(range(0,epochs,1))

  if loss:
    plot_label = 'Loss'
    # Plot validation and training loss
    plt.plot(model_df['loss'],label='Train Loss')
    plt.plot(model_df['val_loss'],label='Validation Loss')

  if acc:
    # Plot validation and training accuracy
    plot_label = 'Accuracy'
    plt.plot(lim, model_df['accuracy'],label='Train Accuracy')
    plt.plot(lim, model_df['val_accuracy'],label='Validation accuracy')

  # set xlabel
  plt.xlabel('Epochs')
  # set ylabel
  plt.ylabel(plot_label)
  # set title
  plt.title(f'Training {plot_label} Vs Validation {plot_label}')
  # set x ticks
  plt.xticks(lim)
  # show legend
  plt.legend()
  plt.show()

def model_scores(deep_model_pred):
  """
  This function converts model probalilities to predicted classes
  params: deep_model_pred: deep_model predictions
  return: box: list of converted predictions
  """
  # Empty list to sore predicitions
  box = []
  # Loop through all probabilities
  for a in deep_model_pred:
    # 1 if probability > 0.5 else 0
    if a[0] > 0.5:
      box.append(1)
    else:
      box.append(0)

  return box

"""$Precision = \frac{TP}{TP + FP}$

$Recall = \frac{TP}{TP + FN}$

$F1 = 2 * \frac{Precision * Recall}{Precision + Recall}$
"""

def eval_scores(actual_vals, pred_vals):
    """
    This function calculates the precision, accuracy, f1-score
    and recall of model predictions on testing data
    actual_vals: Test values
    pred_vals: Predicted values

    return scores: list of all scores
    """
    # Compute F1 score
    f1 = f1_score(y_true=actual_vals, y_pred=pred_vals)
    # Compute Precision
    prec = precision_score(y_true=actual_vals, y_pred=pred_vals)
    # Recall
    recall = recall_score(y_true=actual_vals, y_pred=pred_vals)
    # Accuracy
    accuracy = accuracy_score(y_true=actual_vals, y_pred=pred_vals)
    # Storing all scores in list
    scores = [round(metric,2) for metric in [f1, prec, recall, accuracy]]

    return scores

"""### **Deep Learning Data Preprocessing**"""

# GloVe word embedding
glove_file_path = r'glove.twitter.27B.25d.txt'

# Define data for Deep Learning models
# 10000 rows (5k per class)
small = data.groupby("class").sample(n=5000, random_state=99)
small_text = small['text']
small_class = small['class']

print(f'Length of dataset for deep learning models: {len(small)}')

# Train-test split for Deep Models
deep_train_texts, deep_test_texts, deep_train_class, deep_test_class = train_test_split(small_text, small_class, test_size=0.3, random_state=17)

# Setting 5000 of most frequent tokens to consider
max_vocab=5000

# Initialize tokenizer to obtain total no. of distinct tokens
tokenizer = Tokenizer(num_words = max_vocab)

# Fit Tokenizer on data
tokenizer.fit_on_texts(small_text)

# Set max number of words per sentence
max_len = 200

# Vectorize train and test texts
train_sequences = tokenizer.texts_to_sequences(deep_train_texts)
test_sequences = tokenizer.texts_to_sequences(deep_test_texts)

# Pad or Trim data to defined sentence length
padded_train_seq = keras.utils.pad_sequences(train_sequences, maxlen=max_len)
padded_test_seq = keras.utils.pad_sequences(test_sequences, maxlen=max_len)

# Convert labels to array
model_train_labels = np.array(deep_train_class)
model_test_labels = np.array(deep_test_class)

# Create vocabulary for dataset tokens
deep_dict = {value:key for key,value in tokenizer.word_index.items()}

print(f'There are a total of {len(tokenizer.word_index):,} distinct words in the training data')

# Empty dictionary to store embeddings
embedding_index = {}

# Convert GloVe document to dictionary
with open(glove_file_path, encoding='utf8') as gf:

  # Loop through each line of text
  for each_line in gf:
      word, coeff = each_line.split(maxsplit=1)
      coeff = np.fromstring(coeff, sep=' ')
      embedding_index[word] = coeff

print(f'Glove data loaded. Total Words: {len(embedding_index)}')
print(f'Dimension of Glove data is : {len(embedding_index.get("hello"))}')

num_tokens = len(deep_dict)
# Create embedding matrix
embedding_matrix = np.zeros((num_tokens+2, 25))

# initialize counter to monitor vocabulary values
exist = 0
missing = 0

# Build weights of the embbeddings matrix using GloVe
# Loop though word, value pairs in vocabulary
for vocab_val, word in deep_dict.items():
  embedding_vector = embedding_index.get(word)

  # Mapping words in data vocabulary to embedding
  if embedding_vector is not None:
    embedding_matrix[vocab_val] = embedding_vector
    exist += 1
  else:
    missing += 1

print(f'Total Words: {len(deep_dict)}\nConverted: {exist}\nNot-Converted: {missing}')

"""### **Bi-LSTM MODELS**"""

# Set Early stop criteria
early = keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=3, verbose = 1)

"""#### Bi-LSTM model 1"""

# Input for variable-length sequences of integers
inputs = keras.Input(shape=(None,), dtype="int32")

# Embed each integer in a 128-dimensional vector
x = Embedding(max_vocab, 128)(inputs)

# Add 2 bidirectional LSTMs
x = Bidirectional(LSTM(300, return_sequences=True))(x)
x = Bidirectional(LSTM(350))(x)

# Add a classifier
outputs = Dense(1, activation="sigmoid")(x)
modelx = keras.Model(inputs, outputs)
modelx.summary()

keras.utils.plot_model(modelx)

# Compiling the model
modelx.compile(Adam(learning_rate=0.01), "binary_crossentropy", metrics=["accuracy"])

# Fitting first Bi-LSTM model
hx = modelx.fit(padded_train_seq, model_train_labels, epochs=10, verbose=1,
                              validation_split=0.3, callbacks=early, batch_size = 32)

# Plot Model Perfromance(Accuracy)
eval_deep_model(history=hx, acc=True)

# Predicting with model
dm1_pred = modelx.predict(padded_test_seq)

# Evaluating first Bi-LSTM model
model_eval(pred_vals=model_scores(dm1_pred), actual_vals=deep_test_class)
plt.title('Bi-LSTM1 Predictions')

"""#### Bi-LSTM model 2"""

# Input for variable-length sequences of integers
inputs2 = keras.Input(shape=(None,), dtype="int32")

# Embed each integer in a 128-dimensional vector
x2 = Embedding(input_dim= num_tokens+2, output_dim= 25,
               name='Twitter_25D_Encoding',
               embeddings_initializer=keras.initializers.Constant(embedding_matrix))(inputs2)

# Add 3 bidirectional LSTMs
x2 = Bidirectional(LSTM(300, return_sequences=True))(x2)
x2 = Bidirectional(LSTM(350, return_sequences=True))(x2)
x2 = Bidirectional(LSTM(400, return_sequences=False))(x2)

# Add a classifier
outputs2 = Dense(1, activation="sigmoid")(x2)
modely = keras.Model(inputs2, outputs2)
modely.summary()

# Compile Model
modely.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',
                metrics=['accuracy'])

# Plot model
keras.utils.plot_model(modely)

# Fitting the model
hy = modely.fit(padded_train_seq, model_train_labels, epochs=8, verbose=1,
                              validation_split=0.3, callbacks=early, batch_size = 64)

# Evaluating second Bi-LSTM model
dm2_pred = modely.predict(padded_test_seq)

model_eval(pred_vals=model_scores(dm2_pred), actual_vals=model_test_labels)
plt.title('Bi-LSTM2 Predictions')

# Plot Model Perfromance(Accuracy)
eval_deep_model(history=hy, acc=True)



"""### **GRU MODEL**"""

# Input for variable-length sequences of integers
inputs3 = keras.Input(shape=(None,), dtype="int32")

# Embed each integer in a 25-dimensional vector
x3 = Embedding(input_dim= num_tokens+2, output_dim= 25,
               name='Twitter_25D_Encoding',
               embeddings_initializer=keras.initializers.Constant(embedding_matrix))(inputs3)

# Add 3 GRU
x3 = GRU(300, return_sequences=True, activation='relu')(x3)
x3 = GRU(350, return_sequences=True, activation='relu')(x3)
x3 = GRU(400, return_sequences=False, activation='relu')(x3)

# Add a classifier
outputs3 = Dense(1, activation="sigmoid")(x3)
model_gru = keras.Model(inputs3, outputs3)
model_gru.summary()

# Compile the GRU model
model_gru.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',
                metrics=['accuracy'])

# Plot GRU model architecture
keras.utils.plot_model(model_gru)

# Fitting GRU model
gru_his = model_gru.fit(padded_train_seq, model_train_labels, epochs=10, verbose=1,
                              validation_split=0.3, callbacks=early, batch_size = 64)

# Plot Model Perfromance(Accuracy)
eval_deep_model(history=gru_his, acc=True)

# Evaluating GRU model
dm3_pred = model_gru.predict(padded_test_seq)

# Evaluation metrics
model_eval(pred_vals=model_scores(dm3_pred), actual_vals=model_test_labels)
plt.title('GRU Predictions')

"""# Traditional Model performance on Small dataset"""

# Initizalize the vectorizer
tf_idf_vectorizer2 = TfidfVectorizer(use_idf=True, max_features=500, ngram_range=(1,3), max_df=0.4)

# Fit and transform the training data with the vectorizer
tf_idf_vectorizer2.fit(small['text'])
train_texts_tfidf2 = tf_idf_vectorizer2.transform(deep_train_texts)

# Transform the testing data
test_texts_tfidf2 = tf_idf_vectorizer2.transform(deep_test_texts)

# Taining Logistic model with deep learning data
small_log_model = LogisticRegression(max_iter=1000)
small_log_model.fit(train_texts_tfidf2, deep_train_class)

# using logistic model to predict
deep_logistic_predictions = small_log_model.predict(test_texts_tfidf2)

# Evaluating model
model_eval(actual_vals=deep_test_class, pred_vals=deep_logistic_predictions)
plt.title('Logistic Predictions (DL Data)')

# Train Naive Model using Deep data
small_nb_model = MultinomialNB(alpha=0.6, fit_prior=False)

# Fit Naive model
small_nb_model.fit(train_texts_tfidf2, deep_train_class)

# using Naive model to predict
deep_naive_predictions = small_nb_model.predict(test_texts_tfidf2)

# Evaluating model
model_eval(actual_vals=deep_test_class, pred_vals=deep_naive_predictions)
plt.title('Naive Bayes Predictions (DL Data)')

# Compute Model Scores
scores = {'Bi-LSTM1': eval_scores(actual_vals=deep_test_class, pred_vals=model_scores(dm1_pred)),
          'Bi-LSTM2': eval_scores(actual_vals=deep_test_class, pred_vals=model_scores(dm2_pred)),
          'GRU': eval_scores(actual_vals=deep_test_class, pred_vals=model_scores(dm3_pred)),
          'Logistic': eval_scores(actual_vals=deep_test_class, pred_vals=deep_logistic_predictions),
          'Naive Bayes': eval_scores(actual_vals=deep_test_class, pred_vals=deep_naive_predictions)}

# Display Scores in a Dataframe
pd.DataFrame(scores, index=['F1','Precision','Recall', 'Accuracy'])
