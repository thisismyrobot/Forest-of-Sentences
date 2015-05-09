# -*- coding: utf-8 -*-
""" Worked example of sentence vectorisation using sample data.
"""
import operator
import sklearn.ensemble
import sentence_vectors


print 'Loading training data...',
with open('language_id.txt', 'r') as training_raw_f:
    # csv.reader had real problems with this file?!!?
    data = [map(lambda line: line[1:-1], map(str.strip, line.split(',', 1)))
            for line in
            training_raw_f.readlines()]
print 'Done!'


print 'Preparing vectoriser...',
sentence_vectoriser = sentence_vectors.SentenceVectoriser(
    map(operator.itemgetter(1), data), 200 # 100+ seemed to be about right.
)
print 'Done!'


print 'Preparing training data...',
inputs = [sentence_vectoriser.vector(line[1]) for line in data]
targets = [('English', 'French', 'Spanish').index(line[0]) for line in data]
print 'Done!'


print 'Training RandomForest...',
model = sklearn.ensemble.RandomForestRegressor(n_estimators=10000, n_jobs=-1)
model.fit(inputs, targets)
model.set_params(n_jobs=1) # Otherwise predictions are strangely slow?!?
print 'Done!'


def predict(sentence):
    """ Perform a prediction.

        Return the category of the text and a "percentage" accuracy.
    """
    score = model.predict(sentence_vectoriser.vector(sentence))[0]
    category = round(score, 0)
    return (
        ('English', 'French', 'Spanish')[int(category)],
        int((0.5 - abs(score - category)) * 200)
    )


print 'Testing...\n'
sentences = (
    'Hello mate, how are you? I need to find the toilet.',
    'Bonjour compagnon , comment êtes-vous? Je dois trouver les toilettes .',
    'Hola amigo, ¿cómo estás? Necesito encontrar el inodoro.',
)

for s in sentences:
    print s, predict(s)

print '\nDone!'
