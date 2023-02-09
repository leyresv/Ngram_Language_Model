import pickle
import os
import json

cur_file_path = os.path.dirname(os.path.realpath(__file__))


def load_test_data():
    test_data = json.load(open(os.path.join(cur_file_path, "../../data/regenta_test_data_oov.json")))
    return test_data


def evaluate_dataset_perplexity(dataset, model):
    perp = 0
    len_test_set = len(dataset)
    for sent in dataset:
        sent_perp = model.evaluate_sentence_perplexity(sent)
        if sent_perp is None:
            len_test_set -= 1
        else:
            perp += sent_perp
    return perp / len_test_set


if __name__ == "__main__":
    # Load test data
    test_data = load_test_data()
    # Load trained model
    trigram_lm = pickle.load(open(os.path.join(cur_file_path, "../../models/cervantes_trigram_lm.pkl"), "rb"))
    perplexity = evaluate_dataset_perplexity(test_data, trigram_lm)
    print(perplexity)
