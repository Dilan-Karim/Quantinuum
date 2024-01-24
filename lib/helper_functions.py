import string, re

# Helper functions for loading data
# In the example: https://github.com/CQCL/lambeq/blob/main/docs/examples/tree-reader.ipynb
# The full stop is directly adjacent to the word, which is not the case in the data
# This function removes the whitespace before punctuation while loading the data
def load_data(file_path):
    labels = []
    sentences = []
    with open(file_path, 'r') as file:
        for line in file:
            label, sentence = line.strip().split(' ', 1)
            labels.append(int(label))
            # Remove whitespace before punctuation
            sentence = re.sub(r'\s([{}])'.format(re.escape(string.punctuation)), r'\1', sentence)
            sentences.append(sentence)
    return labels, sentences

# Construct a tree from a sentence
# BobcatParser is a CCG parser
def construct(sentence, reader, plot=True, save_path=None):
    
    diagram = reader.sentence2diagram(sentence=sentence)
    if plot:
        diagram.draw(path=save_path)


    return reader




if __name__=="__main__":
    # path to the data - currently only data within the repo is considered
    path_to_training = "training_data/mc_train_data.txt"
    path_to_test = "training_data/mc_test_data.txt"


    _, sentences = load_data(path_to_training)
    print(sentences)