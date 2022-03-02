import torch
import numpy as np
import pandas as pd
import numpy
import os
import re
import json
import pickle
import tokenizers
import pandas as pd

from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


model = AutoModelForMaskedLM.from_pretrained(
    "Shushant/nepaliBERT", output_hidden_states=True, return_dict=True, output_attentions=True)


tokenizers = AutoTokenizer.from_pretrained("Shushant/nepaliBERT")


def get_bert_embedding_sentence(input_sentence, md=model, tokenizer=tokenizers):
    # md = model
    # tokenizer = tokenizers
    marked_text = " [CLS] " + input_sentence + " [SEP] "
    tokenized_text = tokenizer.tokenize(marked_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)

    # Convert inputs to Pytorch tensors
    tokens_tensors = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        outputs = md(tokens_tensors, segments_tensors)
        # removing the first hidden state
        # the first state is the input state

        hidden_states = outputs.hidden_states
    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding.numpy()


# print(get_bert_embedding_sentence("नेपाल को ससकृती ध्वस्त पार्ने योजना"))

# loading the dataset
df = pd.read_csv('collected_labeled_data.csv')
df = df.drop(df[df['label'] == 2].index)  # dropping neutral sentiments
df.dropna(inplace=True)  # dropping any NA values in dataset

# applying sentence embedding to each texts of the dataset
df['sentence_embeddings'] = df['text'].apply(get_bert_embedding_sentence)

# splitting the data into features and labels
X, y = df['sentence_embeddings'], df['label']
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=420)


# loading the model
svc = SVC()
svc.fit(train_X.tolist(), train_y)

svc_pred = svc.predict(test_X.tolist())

print(f1_score(test_y, svc_pred))


# note 0 is negative and 1 is positive
print(svc.predict(np.array(get_bert_embedding_sentence(
    "नराम्रो कुरा नगरेकै बेश").tolist()).reshape(1, -1)))


print(svc.predict(np.array(get_bert_embedding_sentence(
    "मलाई पढ्न मनपर्छ").tolist()).reshape(1, -1)))
