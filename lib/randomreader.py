"""
This module contains the RandomReader class, which represents a random reader.

The RandomReader class provides methods to convert a sentence into a random diagram
and perform various operations on the diagram.
"""

import random
from lambeq import AtomicType, Reader
from lambeq.backend.grammar import Box, Word

# Define the types
S = AtomicType.SENTENCE # S to be consistent with the BobcatParser
N = AtomicType.NOUN # S to be consistent with the BobcatParser


# Define a new Reader class
# Random composition of words is generated
class RandomReader(Reader):
    """
    A class that represents a random reader.

    This class provides methods to convert a sentence into a random diagram
    and perform various operations on the diagram.

    Attributes:
        None

    Methods:
        sentence2diagram: Converts a sentence into a random diagram.
        converter2word: Converts an object to a Word if it is not already a Word.
        combine_branches: Combines two branches into a UNIBOX.
        recursive_combine: Recursive function to combine adjacent pairs into a UNIBOX.
    """

    def sentence2diagram(self, sentence):
        """
        Converts a sentence into a random diagram.

        Args:
            sentence (str): The sentence to be converted.

        Returns:
            str: The random diagram generated from the sentence.
        """
        words = sentence.split()
        # Shuffle the words to obtain a random diagram
        # Combine the words into a diagram
        diagram = self.recursive_combine(words)
        return diagram

    def converter2word(self, obj):
        """
        Converts an object to a Word if it is not already a Word.

        Args:
            obj: The object to be converted.

        Returns:
            Word: The converted Word object.
        """
        if isinstance(obj, str):
            return Word(obj, N)
        else:
            return obj

    def combine_branches(self, branch1, branch2):
        """
        Combines two branches into a UNIBOX.

        Args:
            branch1: The first branch.
            branch2: The second branch.

        Returns:
            Branch: The combined branch.
        """
        branch1 = self.converter2word(branch1)
        branch2 = self.converter2word(branch2)
        combination = branch1 @ branch2
        new_box = Box("UNIBOX", combination.cod, S)
        combined_branch = combination >> new_box
        return combined_branch

    def recursive_combine(self, items):
        """
        Recursive function to combine adjacent pairs into a UNIBOX.

        Args:
            items (list): The list of items to be combined.

        Returns:
            str: The combined diagram.
        """
        if len(items) <= 1:
            # Base case: if list has 1 or 0 items, just return the list
            return items[0]
        # Shuffle the items to obtain a random diagram
        random.shuffle(items)
        # Combine adjacent pairs
        combined_items = []
        for i in range(0, len(items), 2):
            if i + 1 < len(items):
                combined_items.append(self.combine_branches(items[i], items[i + 1]))
            else:
                # If the list has odd number of elements, add the last element as is
                combined_items.append(items[i])

        # Recursive call
        return self.recursive_combine(combined_items)
