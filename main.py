from lib import helper_functions
from lambeq import BobcatParser, TreeReader, TreeReaderMode
from dataclasses import dataclass

@dataclass
class TreeComparison:
    # initialise the class with input to the data
    path_to_train: str
    path_to_test: str
    # initialise the data by loading it from the path
    def __post_init__(self):
        self.train_labels, self.train_sentences = helper_functions.load_data(self.path_to_train)
        self.test_labels, self.test_sentences = helper_functions.load_data(self.path_to_test)
    def create_lambeq_trees(self, output_path, parser=BobcatParser, treemode=TreeReaderMode.NO_TYPE):
        # create lambeq trees from the sentences
        reader = TreeReader(ccg_parser=parser, mode = treemode)

        # There is a StringBatchType, but it is not well documented
        # With more insight the for loop can be replaced by a batch operation
        trees = list()
        for idx, sentence in enumerate(self.train_sentences):
            trees.append(helper_functions.construct(sentence, reader, save_path=output_path+f"train_{idx}.png"))
        self.lambeq_trees = helper_functions.construct(self.train_sentences[0], reader) 



if __name__ == "__main__":
    # path to the data - currently only data within the repo is considered
    path_to_training = "training_data/mc_train_data.txt"
    path_to_test = "training_data/mc_test_data.txt"


    # initialise the class
    tree_comparison = TreeComparison(path_to_training, path_to_test)

    # create lambeq trees
    tree_comparison.create_lambeq_trees(output_path="./output/BobcatParser/")
