import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import torch
from lambeq import Dataset
from lambeq import PytorchModel, PytorchTrainer
from .helper_functions import print_debug
from .metrics import accuracy, recall, precision, f1

# mplhep settings
plt.style.use(hep.style.CMS)

# Using the builtin binary cross-entropy error from lambeq

class TrainerWrapper:
    """
    Wrapper class for training a quantum model using PyTorch.

    Args:
        model (PytorchModel): The quantum model to be trained.
        comparison_tree (ComparisonTree): The comparison tree
                                          containing the circuits
                                          and labels for training.
        batch_size (int): The batch size for training.
        learning_rate (float): The learning rate for the optimizer.
        epochs (int): The number of training epochs.
        seed (int): The random seed for reproducibility.
        device (int): The device to be used for training.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
    """
    def __init__(self, comparison_tree, batch_size,
                 learning_rate, epochs, seed, debug=False):

        comparison_tree.classical_ansatz()
        print_debug("All circuits created", debug)


        model = PytorchModel.from_diagrams(comparison_tree.get_structured_circuits())
        model_random = PytorchModel.from_diagrams(comparison_tree.get_random_circuits())
        print_debug("Model created", debug)


        # Using the builtin binary cross-entropy error from lambeq
        eval_metrics = {"acc": accuracy,
                        "recall": recall,
                        "precision": precision,
                        "f1": f1}
        print_debug("Loss and metrics created", debug)

        self.train_dataset = Dataset(
                    comparison_tree.train_circuits,
                    comparison_tree.train_labels,
                    batch_size=batch_size)
        
        self.val_dataset = Dataset(
                    comparison_tree.val_circuits,
                    comparison_tree.val_labels,
                    shuffle=False)
        
        self.random_train_dataset = Dataset(
                    comparison_tree.random_train_circuits,
                    comparison_tree.train_labels,
                    batch_size=batch_size)
        
        self.random_val_dataset = Dataset(
                    comparison_tree.random_val_circuits,
                    comparison_tree.val_labels,
                    shuffle=False)
        
        self.test_dataset = Dataset(
                    comparison_tree.test_circuits,
                    comparison_tree.test_labels,
                    shuffle=False)
        
        self.random_test_dataset = Dataset(
                    comparison_tree.random_test_circuits,
                    comparison_tree.test_labels,
                    shuffle=False)
        print_debug("Datasets created", debug)

        self.trainer = PytorchTrainer(
                    model=model,
                    loss_function=torch.nn.BCEWithLogitsLoss(),
                    optimizer=torch.optim.AdamW,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    evaluate_functions=eval_metrics,
                    evaluate_on_train=True,
                    verbose='text',
                    seed=seed)

        print_debug("Trainer created", debug)
        self.trainer_random = PytorchTrainer(
                    model=model_random,
                    loss_function=torch.nn.BCEWithLogitsLoss(),
                    optimizer=torch.optim.AdamW,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    evaluate_functions=eval_metrics,
                    evaluate_on_train=True,
                    verbose='text',
                    seed=seed)
        
        
        print_debug("Random trainer created", debug)
 



    
    def train(self):
        """
        Trains the model using the provided training and validation datasets.

        Returns:
            None
        """
        self.trainer.fit(self.train_dataset, self.val_dataset)

    def train_random(self):
        """
        Trains the random model using the provided training and validation datasets.

        Returns:
            None
        """
        self.trainer_random.fit(self.random_train_dataset, self.random_val_dataset)

    def plot(self):
        fig, axs = plt.subplots(5, 2, sharex="col",sharey="row", figsize=(12,25))  # Adjust to a 5x2 grid layout

        # Set titles for the first row
        axs[0, 0].set_title('BobCat Tree')
        axs[0, 1].set_title('Random Tree')

        # Set labels for the bottom row
        axs[4, 0].set_xlabel('Epochs')
        axs[4, 1].set_xlabel('Epochs')

        # Set labels for the left column
        axs[0, 0].set_ylabel('Loss')
        axs[1, 0].set_ylabel('Accuracy')
        axs[2, 0].set_ylabel('Recall')
        axs[3, 0].set_ylabel('Precision')
        axs[4, 0].set_ylabel('F1 Score')

        range_ = np.arange(1, self.trainer.epochs+1)

        # Plot Loss and Accuracy (existing)
        axs[0, 0].plot(range_, self.trainer.train_epoch_costs, label='train', color='tab:purple')
        axs[0, 0].plot(range_, self.trainer.val_costs, label='val', color='tab:orange')

        axs[1, 0].plot(range_, self.trainer.train_eval_results['acc'], label='train', color='tab:purple')
        axs[1, 0].plot(range_, self.trainer.val_eval_results['acc'], label='val', color='tab:orange')

        axs[0, 1].plot(range_, self.trainer_random.train_epoch_costs, label='train', color='tab:purple')
        axs[0, 1].plot(range_, self.trainer_random.val_costs, label='val', color='tab:orange')

        axs[1, 1].plot(range_, self.trainer_random.train_eval_results['acc'], label='train', color='tab:purple')
        axs[1, 1].plot(range_, self.trainer_random.val_eval_results['acc'], label='val', color='tab:orange')

        # Plot Recall, Precision, and F1 Score (new metrics)
        for metric, row in zip(['recall', 'precision', 'f1'], range(2, 5)):
            axs[row, 0].plot(range_, self.trainer.train_eval_results[metric], label='train', color='tab:purple')
            axs[row, 0].plot(range_, self.trainer.val_eval_results[metric], label='val', color='tab:orange')

            axs[row, 1].plot(range_, self.trainer_random.train_eval_results[metric], label='train', color='tab:purple')
            axs[row, 1].plot(range_, self.trainer_random.val_eval_results[metric], label='val', color='tab:orange')

        # Add legends to the top-left plots
        axs[0, 0].legend()
        axs[0, 1].legend()

        fig.tight_layout()  # Adjust layout to prevent overlap
        fig.savefig('./output/plots/training.pdf')



