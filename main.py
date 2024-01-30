"""
This script performs tree generation, training, and plotting
for the comparison of the BobCat and Random parsers.
"""
from lambeq import TreeReaderMode
from lib import treegenerator
from lib.helper_functions import print_debug
from lib.training import TrainerWrapper

# set debug flag
DEBUG = True

# hyperparameters
BATCH_SIZE = 4
EPOCHS = 200
LEARNING_RATE = 2e-3
SEED = 0
READERMODE = TreeReaderMode.RULE_ONLY

# paths to data
PATH_TO_TRAINING = "training_data/mc_train_data.txt"
PATH_TO_TEST = "training_data/mc_test_data.txt"
PATH_TO_VAL = "training_data/mc_dev_data.txt"

DATA_PATHS = [PATH_TO_TRAINING, PATH_TO_TEST, PATH_TO_VAL]

# output paths
OUTPUT_PATH_BOBCAT = "./output/BobcatParser/"
OUTPUT_PATH_RANDOM = "./output/RandomParser/"
DRAW = True

# initialise the class
tree_comparison = treegenerator.TreeComparison(DATA_PATHS)
print_debug("Data loaded", DEBUG)
# create random trees
tree_comparison.create_lambeq_trees(treemode=READERMODE, output_path=OUTPUT_PATH_BOBCAT, draw=DRAW)
print_debug("Lambeq trees created", DEBUG)
tree_comparison.create_random_trees(output_path=OUTPUT_PATH_BOBCAT, draw=DRAW)
print_debug("Random trees created", DEBUG)
# prepare and train the model
trainer = TrainerWrapper(tree_comparison, BATCH_SIZE, LEARNING_RATE, EPOCHS, SEED, DEBUG)
trainer.train()
trainer.train_random()
# plot the results
trainer.plot()
