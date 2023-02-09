import json
import os
import pickle
import random
from datasets import load_dataset
from collections import Counter
from spacy.lang.es import Spanish

cur_file_path = os.path.dirname(os.path.realpath(__file__))


def tokenize_dataset(data, eos_token="<EOS>"):
    nlp = Spanish()
    nlp.add_pipe("sentencizer")
    tok_sentences = []
    for i, entry in enumerate(data):
        if i % 10000 == 0:
            print(i)
        if not entry["title"].isdigit():
            text = entry["text"].replace("\n", "").replace("'", "").replace('"', "").replace("-", "")
            sentences = nlp(text)
            for sentence in sentences.sents:
                sentence = sentence.text.strip().lower()
                tok_sentence = nlp(sentence)
                if len(tok_sentence) > 10:
                    tok_sentences.append([token.text for token in tok_sentence] + [eos_token])
    print("Tokenization finished")
    return tok_sentences


def tokenize_cervantes_dataset(dataset, eos_token="<EOS>"):
    with open(dataset, "r") as f:
        data = f.readlines()

    # Get lines and remove unwanted character
    data_lst = [line.strip() for line in data]
    data_lst = [line.replace("\ufeff", "").replace("-", "")  for line in data_lst if line!=""]

    # Join all lines to get the whole text
    data_txt = " ".join(data_lst)

    # Identify sentences in text
    nlp = Spanish()
    nlp.add_pipe("sentencizer")
    tok_sentences = []
    sentences = nlp(data_txt)
    for sentence in sentences.sents:
        if len(sentence) > 10 and len(sentence) < 200:
            tok_sentences.append([token.text.lower() for token in sentence] + [eos_token])

    return tok_sentences


def create_vocab(sentences, min_freq=3, unk_token="<UNK>"):
    counts = Counter(word for sentence in sentences for word in sentence)
    vocab = {unk_token: 0}
    for word, count in counts.items():
        if count >= min_freq:
            vocab[word] = count
        else:
            vocab[unk_token] += count
    json.dump(vocab, open(os.path.join(cur_file_path, "../../data/cervantes_vocab_counts.json"), 'w'))
    print("Vocab created")
    return vocab


def preprocess_data(data, vocab, unk_token="<UNK>"):
    sentences = []
    for sentence in data:
        sentences.append([token if token in vocab else unk_token for token in sentence])
    print("Data preprocessed")
    return sentences


if __name__ == "__main__":

    train_dataset = os.path.join(cur_file_path, "../../data/cervantes_train.txt")
    test_dataset = os.path.join(cur_file_path, "../../data/cervantes_test.txt")

    train_data = tokenize_cervantes_dataset(train_dataset)
    test_data = tokenize_cervantes_dataset(test_dataset)
    json.dump(train_data, open(os.path.join(cur_file_path, "../../data/cervantes_train_data.json"), 'w'))
    json.dump(test_data, open(os.path.join(cur_file_path, "../../data/cervantes_test_data.json"), 'w'))

    vocab = create_vocab(train_data)

    train_data_oov = preprocess_data(train_data, vocab)
    test_data_oov = preprocess_data(test_data, vocab)
    json.dump(train_data_oov, open(os.path.join(cur_file_path, "../../data/cervantes_train_data_oov.json"), 'w'))
    json.dump(test_data_oov, open(os.path.join(cur_file_path, "../../data/cervantes_test_data_oov.json"), 'w'))

    regenta = os.path.join(cur_file_path, "../../data/regenta.txt")
    test_data = tokenize_cervantes_dataset(regenta)
    json.dump(regenta, open(os.path.join(cur_file_path, "../../data/regenta_test_data.json"), 'w'))
    vocab = json.load(open(os.path.join(cur_file_path, "../../data/cervantes_vocab_counts.json")))
    test_data_oov = preprocess_data(test_data, vocab)
    json.dump(test_data_oov, open(os.path.join(cur_file_path, "../../data/regenta_test_data_oov.json"), 'w'))







