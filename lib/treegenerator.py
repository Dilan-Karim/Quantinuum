"""
This module contains the TreeComparison class which is used to compare and
generate lambeq trees and random trees.
"""

from dataclasses import dataclass
import helper_functions
from random_reader import RandomReader
from lambeq import BobcatParser, TreeReader, TreeReaderMode

@dataclass
class TreeComparison:
    """
    Class for comparing and generating lambeq trees and random trees.
    """

    def __init__(self, data_paths: list[str]):
        """
        Initializes the TreeComparison class.

        Args:
            data_paths (list[str]): List of paths to the data files.

        Returns:
            None
        """
        self.data_paths = data_paths


    def __post_init__(self):
        """
        Initializes the data by loading it from the path.

        Returns:
            None
        """
        self.train_labels, self.train_sentences = helper_functions.load_data(self.data_paths[0])
        self.test_labels, self.test_sentences = helper_functions.load_data(self.data_paths[1])
        self.val_labels, self.val_sentences = helper_functions.load_data(self.data_paths[2])

    def create_lambeq_trees(self, output_path: str, parser=BobcatParser, treemode=TreeReaderMode.NO_TYPE, draw=False):
        """
        Creates lambeq trees from the sentences.

        Args:
            output_path (str): The path to save the generated trees.
            parser (class, optional): The parser class to use for parsing the sentences. Defaults to BobcatParser.
            treemode (int, optional): The mode for tree reading. Defaults to TreeReaderMode.NO_TYPE.
            draw (bool, optional): Whether to draw the trees. Defaults to False.

        Returns:
            None
        """
        reader = TreeReader(ccg_parser=parser, mode=treemode)

        self.ccg_trees_train = self.collection2tree(self.train_sentences, reader, output_path, "train", draw)
        self.ccg_trees_test = self.collection2tree(self.test_sentences, reader, output_path, "test", draw)
        self.ccg_trees_val = self.collection2tree(self.val_sentences, reader, output_path, "val", draw)

        return None

    def create_random_trees(self, output_path=None, draw=False):
        """
        Creates random trees based on the given sentences and saves them to the specified output path.

        Args:
            output_path (str, optional): The path where the generated trees will be saved. Defaults to None.
            draw (bool, optional): Whether to draw the generated trees. Defaults to False.

        Returns:
            None
        """
        reader = RandomReader()

        self.random_trees_train = self.collection2tree(self.train_sentences, reader, output_path, "train", draw)
        self.random_trees_test = self.collection2tree(self.test_sentences, reader, output_path, "test", draw)
        self.random_trees_val = self.collection2tree(self.val_sentences, reader, output_path, "val", draw)

    def collection2tree(self, collection, reader, output_path, name, draw):
        """
        Converts a collection of sentences into a list of trees.

        Args:
            collection (list): A collection of sentences.
            reader (Reader): An instance of the Reader class.
            output_path (str): The path to save the generated trees.
            name (str): The name prefix for the saved tree images.
            draw (bool): Flag indicating whether to draw the trees.

        Returns:
            list: A list of trees generated from the sentences.
        """
        list_of_trees = list()
        # create trees from the sentences
        for idx, sentence in enumerate(collection):
            list_of_trees.append(helper_functions.construct(sentence, reader, save_path=output_path+f"{name}_{idx}.png", draw=draw))
        return list_of_trees

if __name__ == "__main__":
    # path to the data - currently only data within the repo is considered
    PATH_TO_TRAINING = "training_data/mc_train_data.txt"
    PATH_TO_TEST = "training_data/mc_test_data.txt"
    PATH_TO_VAL = "training_data/mc_dev_data.txt"

    PATH_BOBCATPARSER = "./output/BobcatParser/"
    PATH_RANDOMPARSER = "./output/RandomParser/"

    DATA_PATHS = [PATH_TO_TRAINING, PATH_TO_TEST, PATH_TO_VAL]

    # initialise the class
    tree_comparison = TreeComparison(DATA_PATHS)

    # create random trees
    tree_comparison.create_lambeq_trees(output_path=PATH_BOBCATPARSER, draw=False)
    tree_comparison.create_random_trees(output_path=PATH_RANDOMPARSER, draw=False)
