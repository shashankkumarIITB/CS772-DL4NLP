# CS772 - DL4NLP
## Assignment 1 (Sentiment Analysis)

## Group 24

The submission consists of 6 python files, 5 folders and 1 readme file.

### Python files
| File name | Description |
| ------ | ------ |
| balance_train.&#8203;py | Solve data imbalance in training dataset using oversampling |
| neuralnet.&#8203;py | Contains the model and other related functions to train and test |
| preprocess.&#8203;py | Contains different functions for preprocessing alongwith the embeddings |
| server.&#8203;py | Contains the code for GUI (web app)  |
| test.&#8203;py | Contains the code for testing the model on test datatset |
| train.&#8203;py | Contains the code to train and generate the model |

### Folders
| Folder name | Description |
| ------ | ------ |
| data | Contains the test and train dataset |
| models | Contains the model generated after training |
| output | Contains the predictions |
| slide | Contains the slide for the presentation |
| templates | contains the html file for GUI testing |

## Execution
Choose the required embedding from the dictionary in line 7 of preprocess.&#8203;py 
> embedding_models = {'word2vec': 'word2vec-google-news-300', 'fasttext': 'fasttext-wiki-news-subwords-300', 'glove': 'glove-wiki-gigaword-300'}

and update the key value in line 10 in the same file.
> path = api.load(embedding_models['word2vec'], return_path=True)

Then save and run preprocess.&#8203;py.

```sh
python3 preprocess.py
```

This will download and save the embedding for further use.

Next run the balance_train.&#8203;py to balance the training dataset (data/train.&#8203;csv).

```sh
python3 balance_train.py
```
This will create a new file (data/train_balanced.&#8203;csv), which will be used for training.

Next run the train.&#8203;py to train and generate the model, which will be saved in models/Assignment2.&#8203;h5.

```sh
python3 train.py
```

Then run the test.&#8203;py to test the model on the test dataset (data/gold_test.&#8203;csv).

```sh
python3 test.py
```

## GUI testing
> Note: Download the trained models from https://drive.google.com/drive/folders/1qfgY73JQ9Q4jlGT0vdyZNgpaRoURny_J and save them to models folder for this testing.

Then, run the following commands in the terminal.
```sh
export FLASK_APP=server.py
flask run
```

Once the webpage is live in terminal, open http://localhost:5000/ in the browser.
Then write the sentence for the prediction, select the activation and embedding to use, and click on predict button. It will show the predicted class, alongwith the probabilities.