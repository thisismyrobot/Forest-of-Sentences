# Forest of sentences

An attempt to train scikit-learn's RandomForest clevers with sentence-based
data, as in: https://cloud.google.com/prediction/docs/hello_world

Ignoring the training, this code is really a proof-of-concept attempt at
creating vectors from strings.

## Algorithm

Both the RandomForestRegressor and RandomForestClassifier obviously work fine
with float training values but won't handle strings out of the box (unlike
Google's Prediction API).

To create training vectors from sentences, I first create a collection (100 in
this example) of random "landmark" sentences from the training data. The
fuzzywuzzy 'token\_set_ratio' is then used to calculate a "distance" between
each of the sentences in the training set and the landmarks. These landmark
distances are then used to train the sklearn.ensemble.RandomForestRegressor
against the categories that these sentences fall in to.

I could train a RandomForestClassifier with this same data, but I have been
working heavily with the Regressor recently so I'm sticking with that :)

## Requires

    fuzzywuzzy
    scikit-learn (and all its dependencies etc)

I strongly recommend also installing python-Levenshtein for a 10x speedup in
training.

## Example run

Translations from English to French and Spanish using Google Translate so I'm
sorry if my example sentences are actually quite rude in those languages...

    python forest_of_sentences.py

    Loading training data... Done!
    Preparing vectoriser... Done!
    Preparing training data... Done!
    Training RandomForest... Done!
    Testing...

    Hello mate, how are you? I need to find the toilet. ('English', 99)
    Bonjour compagnon , comment êtes-vous? Je dois trouver les toilettes . ('French', 87)
    Hola amigo, ¿cómo estás? Necesito encontrar el inodoro. ('Spanish', 89)

    Done!

## TODO

Err, make this code less of a procedural stream-of-conscience...
