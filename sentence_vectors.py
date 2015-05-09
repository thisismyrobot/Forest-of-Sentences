""" Sentence vectorisation using landmarks.
"""
import fuzzywuzzy.fuzz
import random


def create_landmarks(source_data, n_landmarks):
    """ Returns a set of landmarks.

        Based on source data that is a list of sentences.
    """
    # Pick a random selection of lines from the source data
    return random.sample(source_data, max(len(source_data), n_landmarks))


def vector(sentence, landmarks):
    """ Return a vector for a sentence, based on landmarks.
    """
    length = float(len(sentence))

    # Should provide a performance improvement.
    ratio_func = fuzzywuzzy.fuzz.token_set_ratio

    return [ratio_func(sentence, landmark) / length
            for landmark
            in landmarks]


class SentenceVectoriser(object):
    """ Manages the vectorisation of sentences.
    """
    def __init__(self, source_data, n_landmarks=10):
        self._landmarks = create_landmarks(source_data, n_landmarks)

    @property
    def landmarks(self):
        """ Return a copy of the current landmarks.
        """
        return self._landmarks[:]

    def vector(self, sentence):
        """ Return a vector based on a sentence.
        """
        return vector(sentence, self._landmarks)
