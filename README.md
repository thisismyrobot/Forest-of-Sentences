# Forest of sentences

An attempt to use Random Forests for https://cloud.google.com/prediction/docs/hello_world

Uses fuzzywuzzy to calculate distances to a set of "goal post" sentences, this
becomes the vector representation of the sentence.

## Requires

    fuzzywuzzy
    scikit-learn (and all its dependencies etc)

## Example run

Please excuse the utf-8 mangling of a windows console...

    python forest_of_sentences.py

    Loading training data... Done!
    Preparing vectoriser... Done!
    Preparing training data... Done!
    Training RandomForest... Done!
    Testing...

    Hello mate, how are you? I need to find the toilet. ('English', 99)
    Bonjour compagnon , comment ├¬tes-vous? Je dois trouver les toilettes . ('French', 87)
    Hola amigo, ┬┐c├│mo est├ís? Necesito encontrar el inodoro. ('Spanish', 89)

    Done!
