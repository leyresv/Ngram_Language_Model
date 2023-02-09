import json
import os
import pickle

from src.model.trigram_language_model import TrigramLM

cur_file_path = os.path.dirname(os.path.realpath(__file__))


def load_train_data():
    train_data = json.load(open(os.path.join(cur_file_path, "../../data/cervantes_train_data_oov.json")))
    vocab = json.load(open(os.path.join(cur_file_path, "../../data/cervantes_vocab_counts.json")))

    return train_data, vocab


if __name__ == "__main__":
    # Load train data
    train_data, vocab = load_train_data()
    # Train trigram language model
    trigram_lm = TrigramLM()
    trigram_lm.train(train_data, vocab)
    # Save LM
    pickle.dump(trigram_lm, open(os.path.join(cur_file_path, "../../models/cervantes_trigram_lm.pkl"), "wb"))