"""
This lib.randomreaderns the TreeComparison class which is used to compare and
generate lambeq trees and random trees.
"""

from dataclasses import dataclass
from lambeq import BobcatParser, TreeReader, TreeReaderMode
from lambeq import AtomicType, TensorAnsatz, RemoveCupsRewriter
from lambeq.backend.tensor import Dim
from lib import helper_functions
from lib.randomreader import RandomReader

@dataclass
class TreeComparison:
    """
    Class for comparing and generating lambeq trees and random trees.
    """
    data_paths: list[str]

    def __post_init__(self):
        """
        Initializes the data by loading it from the path.

        Returns:
            None
        """
        self.train_labels, self.train_sentences = helper_functions.load_data(self.data_paths[0])
        self.test_labels, self.test_sentences = helper_functions.load_data(self.data_paths[1])
        self.val_labels, self.val_sentences = helper_functions.load_data(self.data_paths[2])

    def create_lambeq_trees(self,
                            output_path = None,
                            parser=BobcatParser,
                            treemode=TreeReaderMode.NO_TYPE,
                            draw=False):
        """
        Creates lambeq trees from the sentences.

        Args:
            output_path (str): The path to save the generated trees.
            parser (class, optional): The parser class. Defaults to BobcatParser.
            treemode (int, optional): The mode for tree reading. Defaults to TreeReaderMode.NO_TYPE.
            draw (bool, optional): Whether to draw the trees. Defaults to False.

        Returns:
            None
        """
        reader = TreeReader(ccg_parser=parser, mode=treemode)

        self.ccg_trees_train = self.collection2tree(self.train_sentences,
                                                    reader,
                                                    output_path,
                                                    "train",
                                                    draw)

        self.ccg_trees_test = self.collection2tree(self.test_sentences,
                                                   reader,
                                                   output_path,
                                                   "test",
                                                   draw)

        self.ccg_trees_val = self.collection2tree(self.val_sentences,
                                                  reader,
                                                  output_path,
                                                  "val",
                                                  draw)

        return None

    def create_random_trees(self, output_path=None, draw=False):
        """
        Creates random trees based on the given sentences and
        saves them to the specified output path.

        Args:
            output_path (str, optional): The path where the trees will be saved. Defaults to None.
            draw (bool, optional): Whether to draw the generated trees. Defaults to False.

        Returns:
            None
        """
        reader = RandomReader()

        self.random_trees_train = self.collection2tree(self.train_sentences,
                                                       reader,
                                                       output_path,
                                                       "train",
                                                       draw)

        self.random_trees_test = self.collection2tree(self.test_sentences,
                                                      reader,
                                                      output_path,
                                                      "test",
                                                      draw)

        self.random_trees_val = self.collection2tree(self.val_sentences,
                                                     reader,
                                                     output_path,
                                                     "val",
                                                     draw)

    def collection2tree(self, collection, reader, output_path = None, name = None, draw = False):
        """
        Converts a collection of sentences into a list of trees.

        Args:
            collection (list): A collection of sentences.
            reader (Reader): An instance of the Reader class.
            output_path (str, optional): The path to save the generated trees. Defaults to None.
            name (str, optional): The name prefix for the saved tree images. Defaults to None.
            draw (bool, optional): Flag indicating whether to draw the trees. Defaults to False.

        Returns:
            list: A list of trees generated from the sentences.
        """
        list_of_trees = list()
        # create trees from the sentences
        for idx, sentence in enumerate(collection):
            if output_path and name:
                list_of_trees.append(helper_functions.construct(sentence,
                                                                reader,
                                                                save_path=output_path+
                                                                          f"{name}_{idx}.png",
                                                                draw=draw))
            else:
                list_of_trees.append(helper_functions.construct(sentence, reader))
        return list_of_trees

    def classical_ansatz(self):
        """
        Creates a classical ansatz for the given number of qubits and layers.

        Returns:
            Ansatz: The classical ansatz.
        """
        N = AtomicType.NOUN
        S = AtomicType.SENTENCE
        ansatz = TensorAnsatz({S: Dim(2), N: Dim(2)})

        remove_cups = RemoveCupsRewriter()

        self.train_circuits = [ansatz(remove_cups(tree)) for tree in self.ccg_trees_train]
        self.test_circuits = [ansatz(remove_cups(tree)) for tree in self.ccg_trees_test]
        self.val_circuits = [ansatz(remove_cups(tree)) for tree in self.ccg_trees_val]

        self.random_train_circuits = [ansatz(remove_cups(tree)) for tree in self.random_trees_train]
        self.random_test_circuits = [ansatz(remove_cups(tree)) for tree in self.random_trees_test]
        self.random_val_circuits = [ansatz(remove_cups(tree)) for tree in self.random_trees_val]

    def get_structured_circuits(self):
        """
        Returns all the circuits.

        Returns:
            list: A list of all the circuits.
        """
        return self.train_circuits + self.test_circuits

    def get_random_circuits(self):
        """
        Returns all the random circuits.

        Returns:
            list: A list of all the random circuits.
        """
        return self.random_train_circuits + self.random_test_circuits
