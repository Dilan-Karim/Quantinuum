"""
This module contains helper functions for loading data and constructing diagrams.

Functions:
    - load_data(filename): Load data from a file.
    - construct(sentence, reader, save_path=None, draw=False): Constructs a diagram based 
                                                               on the given sentence using 
                                                               the provided reader.

Classes:
    - Reader: A class used to convert sentences into diagrams.

"""

def load_data(filename):
    """
    Load data from a file.

    Args:
        filename (str): The path to the file containing the data.

    Returns:
        tuple: A tuple containing two lists - labels and sentences.
               The labels list contains pairs of floats representing the labels,
               and the sentences list contains the corresponding sentences.
    """
    labels, sentences = [], []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            t = float(line[0])
            labels.append([t, 1-t])
            sentences.append(line[1:].strip())
    return labels, sentences


def construct(sentence, reader, save_path=None, draw=False):
    """
    Constructs a diagram based on the given sentence using the provided reader.

    Args:
        sentence (str): The sentence to be converted into a diagram.
        reader (Reader): The reader object used to convert the sentence into a diagram.
        save_path (str, optional): The path to save the diagram image file. Defaults to None.
        draw (bool, optional): Flag indicating whether to draw the diagram. Defaults to False.

    Returns:
        Diagram: The constructed diagram object.
    """
    diagram = reader.sentence2diagram(sentence=sentence)
    if draw:
        if save_path:
            diagram.draw(path=save_path)
        else:
            print("No save path provided. Generating diagram without saving.")

    return diagram
