# CS772 - DL4NLP
## Assignment 3

## Group 24

The submission consists of 1 pdf file (presentation), 2 folders and 1 readme file.

### Folders
| Folder name | Description |
| ------ | ------ |
| WITH Embedding | Contains the code with pre-trained (glove) embedding |
| WITHOUT Embedding | Contains the code with no pre-trained embedding |

Every folder contains 5 python files and 3 subfolders

### Python files
| File name | Description |
| ------ | ------ |
| neuralnet.&#8203;py | Contains the model and other related functions to train and test |
| preprocess.&#8203;py | Contains different functions for preprocessing (alongwith the embeddings) |
| server.&#8203;py | Contains the code for GUI (web app)  |
| test.&#8203;py | Contains the code for testing the model on test datatset |
| train.&#8203;py | Contains the code to train and generate the model |

### Subfolders
| Subfolder name | Description |
| ------ | ------ |
| data | Contains the data for training and testing |
| models | Contains the model (after training) |
| templates | contains the html file for GUI testing |

## Execution
Run preprocess.&#8203;py.

```sh
python3 preprocess.py
```

In pre-trained embedding case, this will also download and save the embedding for further use.

Next select the model to be trained in train.&#8203;py file. Value guide for different models is given below -

| bi value | Model |
| ------ | ------ |
| 0 | LSTM |
| 1 | GRU |
| 2 | RNN |
| 3 | Bi-LSTM |
| 4 | Bi-GRU |

| ci value | Model |
| ------ | ------ |
| 5 | single model layer |
| bi (0,1,2,3,4) | double model layer |

| di value | Model |
| ------ | ------ |
| 0 | no hidden layer |
| 1 | one hidden layer |

(All the possible combination are given in the list l.)
Next run the train.&#8203;py to train and generate the model, which will be saved in models/bi-ci-di.&#8203;h5 according to the values chosen. By default, train.&#8203;py will generate 0-5-0 model, i.e., model with 1 LSTM and no hidden layer.

```sh
python3 train.py
```

Then select the model to test in test.&#8203;py and run the file to test the model on the test dataset (data/gold_test.&#8203;csv). By default, test.&#8203;py will test 0-5-0 model, i.e., model with 1 LSTM and no hidden layer.

```sh
python3 test.py
```

## GUI testing
> Note: Download the trained models from https://drive.google.com/drive/folders/14yLseyRRrqKfOwxOpVFQjqJSK9kK6s6F?usp=sharing and save them to respective models folder for this testing.

Then, run the following commands in the terminal.
```sh
export FLASK_APP=server.py
flask run
```

Once the webpage is live in terminal, open http://localhost:5000/ in the browser.
Then write the sentence for the prediction, select the model to use, and click on predict button. It will show the predicted class, alongwith the probabilities.