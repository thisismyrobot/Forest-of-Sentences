# -*- coding: utf-8 -*-
""" The prediction model module.
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
    map(operator.itemgetter(1), data), 100
)
print 'Done!'


print 'Preparing training data...',
inputs = []
targets = []
for lang, line in data:
    inputs.append(sentence_vectoriser.vector(line))
    targets.append(('English', 'French', 'Spanish').index(lang))
print 'Done!'


print 'Training RandomForest...',
model = sklearn.ensemble.RandomForestRegressor(n_estimators=10000, n_jobs=-1)
model.fit(inputs, targets)
print 'Done!'


def predict(sentence):
    """ Perform a prediction.
    """
    score = model.predict(sentence_vectoriser.vector(sentence))[0]
    cat = round(score, 0)
    return ('English', 'French', 'Spanish')[int(cat)], int((0.5 - abs(score - cat)) * 200)


print 'Testing...\n'
sentences = (
    'Hello mate, how are you? I need to find the toilet.',
    'Bonjour compagnon , comment êtes-vous? Je dois trouver les toilettes .',
    'Hola amigo, ¿cómo estás? Necesito encontrar el inodoro.',
)

for sentence in sentences:
    print sentence, predict(sentence)

print '\nDone!'
