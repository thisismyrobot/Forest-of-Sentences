""" Sentence vectorisation using landmarks.
"""
import fuzzywuzzy.fuzz
import random


def create_goal_posts(source_data, n_goal_posts=10):
    """ Returns a set of goal posts.

        Based on source data that is a list of sentences.
    """
    # Pick a selection of lines from the source data
    examples = [random.choice(source_data).split(' ')
                for _goal in xrange(n_goal_posts)]
    map(random.shuffle, examples)
    examples = map(' '.join, examples)
    return examples


def vector(sentence, goal_posts):
    """ Return a vector for a sentence, based on goal posts.
    """
    length = float(len(sentence))
    return [fuzzywuzzy.fuzz.token_set_ratio(sentence, goal_post) / length
            for goal_post
            in goal_posts]


class SentenceVectoriser(object):
    """ Manages the vectorisation of sentences.
    """
    def __init__(self, source_data, n_goal_posts=10):
        self._goal_posts = create_goal_posts(source_data, n_goal_posts)

    def vector(self, sentence):
        """ Return a vector based on a sentence.
        """
        return vector(sentence, self._goal_posts)
