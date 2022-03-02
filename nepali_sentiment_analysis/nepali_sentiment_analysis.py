import os

import torch
import numpy as np
import pandas as pd

import pickle
import pandas as pd

from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")


model = AutoModelForMaskedLM.from_pretrained(
    "Shushant/nepaliBERT", output_hidden_states=True, return_dict=True, output_attentions=True)


tokenizers = AutoTokenizer.from_pretrained("Shushant/nepaliBERT")

root_dir = os.path.expanduser('~/')


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

def predict(model_name, sentence):
    # loading the model
    model_path = os.path.join(root_dir,'nepali_sentiment', model_name)
    model = pickle.load(open(model_path, "rb"))

    # note 0 is negative and 1 is positive
    print(model.predict(np.array(get_bert_embedding_sentence(
        sentence).tolist()).reshape(1, -1))[0])
