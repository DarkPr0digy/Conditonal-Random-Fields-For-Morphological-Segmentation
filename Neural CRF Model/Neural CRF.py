#from bilstm_crf import BiLSTM_CRF
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn_crfsuite import metrics
from bi_lstm_crf.app import train, WordsTagger
#from .bi_lstm_crf.app import train, WordsTagger
import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


class NeuralCRF:
    def __init__(self, language: str):
        self.language = language
        self.input_files = ["../morphology/" + language + '/' + language + ".clean.train.conll",
                            "../morphology/" + language + '/' + language + ".clean.dev.conll",
                            "../morphology/" + language + '/' + language + ".clean.test.conll"]
        self.labels = []

    def surface_segmentation(self):
        # Collect Data
        ####################################################
        training_data, dev_data, test_data = [], [], []
        lists = (training_data, dev_data, test_data)
        datas_set = open(os.path.join(sys.path[0], 'corpus_dir/dataset.txt'), "w")
        counter = 0
        for file in self.input_files:

            input_file = open(os.path.join(sys.path[0], file), 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")
                word = list(content[0])
                segments = content[1].split('-')
                label = ""
                for morph in segments:
                    if len(morph) == 1:
                        label += "S"
                    else:
                        label += "B"
                        for i in range(len(morph) - 2):
                            label += "M"
                        label += "E"

                string = []
                for morph in segments:
                    if len(morph) == 1:
                        string.append("S")
                    else:
                        string.append("BW")
                        for i in range(len(morph) - 2):
                            string.append("M")
                        string.append("E")

                string = format_arrays_json(string)
                tmp = (word, label)
                lists[counter].append(tmp)
                if counter == 0:
                    datas_set.write("".join(word) + "\t" + string + "\n")
            input_file.close()
            counter += 1

        datas_set.close()
        print("Collected Data")


        vocab = open(os.path.join(sys.path[0], 'corpus_dir/vocab.json'), "w")
        tags = open(os.path.join(sys.path[0], 'corpus_dir/tags.json'), "w")

        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\'.\",1234567890%"
        string = "["
        count = 0
        for char in alphabet:
            if char == "\"":
                string += "\"\\" + char + "\""
            else:
                string += "\"" + char + "\""
            if count != len(alphabet)-1:
                string += ", "
            count += 1
        string += "]"
        #vocab.write(str(list(alphabet)))
        vocab.write(string)
        vocab.close()

        word_to_ix = {}
        for letters in alphabet:
            word_to_ix[letters] = len(word_to_ix)
        tag_to_ix = {"S": 0, "B": 1, "M": 2, "E": 3}
        #tags.write("[\"S\", \"B\", \"M\", \"E\"]")
        tags.write("[\"S\", \"BW\", \"M\", \"E\"]")
        tags.close()

        embedding_dim = 5
        hidden_dim = 4

        ####################################################################################

        args = get_surface_args()
        print("Got Arguments")
        train(args)
        print("Completed Training")
        df = pd.read_csv("surface_model/loss.csv")
        df[["train_loss", "val_loss"]].ffill().plot(grid=True)
        plt.show()
        temp_predicted = []
        temp_true = []
        model = WordsTagger(model_dir='surface_model')
        print("Testing Model")
        for word, label in test_data:
            temp_true.append(label)
            tmp = model([''.join(word)], begin_tags="BS")[0]
            temp_predicted.append(tmp)

        y_true = []

        for lbl in temp_true:
            y_true.append(list(lbl))

        print(y_true[0:15])

        y_predicted = []
        for arr in temp_predicted:
            tmp = arr[0]
            str = []
            for char in tmp:
                if char == "BW":
                    str.append("B")
                else:
                    str.append(char)
            y_predicted.append(str)
        print(y_predicted[0:15])

        return y_predicted, y_true


    def surface_labelled_segmentation(self):
        # Collect Data
        ####################################################
        training_data, dev_data, test_data = [], [], []
        lists = (training_data, dev_data, test_data)
        counter = 0
        alphabet, labels = [], []
        datas_set = open(os.path.join(sys.path[0], 'corpus_dir/dataset.txt'), "w")

        for file in self.input_files:
            input_file = open(os.path.join(sys.path[0], file), 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")

                surface_form = content[1].split('-')
                label_list = get_ortho_labels(content[2])

                for segment in surface_form:
                    if segment not in alphabet:
                        alphabet.append(segment)
                for label in label_list:
                    if label not in labels:
                        labels.append(label)

                tmp = (surface_form, label_list)
                lists[counter].append(tmp)

                tmp_surface = format_arrays_json(surface_form)

                tmp_labels = format_arrays_json(label_list)

                if counter == 0:
                    datas_set.write(tmp_surface + "\t" + tmp_labels + "\n")

            input_file.close()
            counter += 1

        datas_set.close()
        print("Collected Data: labels and segments")
        ##############################################
        vocab = open(os.path.join(sys.path[0], 'corpus_dir/vocab.json'), "w")
        tags = open(os.path.join(sys.path[0], 'corpus_dir/tags.json'), "w")

        tmp_alphabet = format_arrays_json(alphabet)
        tmp_labels = format_arrays_json(labels)
        self.labels = labels

        vocab.write(tmp_alphabet)
        vocab.close()

        tags.write(tmp_labels)
        tags.close()

        args = get_labelled_args()
        print("Got Arguments")

        train(args)
        print("Completed Training")

        df = pd.read_csv("labelled_model/loss.csv")
        df[["train_loss", "val_loss"]].ffill().plot(grid=True)
        plt.show()

        model = WordsTagger(model_dir='labelled_model')
        print("Testing Model")

        y_true = []
        y_predicted = []
        for segments, labels in test_data:
            y_true.append(labels)

            tmp = model([segments])[0][0]
            y_predicted.append(tmp)

        return y_predicted, y_true

    def results(self, Y_predict, Y_test):
        test = MultiLabelBinarizer().fit_transform(Y_test)
        predicted = MultiLabelBinarizer().fit_transform(Y_predict)

        print('Weighted:')
        print("Recall: " + str(recall_score(test, predicted, average='weighted')))
        print("Precision: " + str(precision_score(test, predicted, average='weighted')))
        print("F1 Score: " + str(f1_score(test, predicted, average='weighted')))
        print('Micro:')
        print("Recall: " + str(recall_score(test, predicted, average='micro')))
        print("Precision: " + str(precision_score(test, predicted, average='micro')))
        print("F1 Score: " + str(f1_score(test, predicted, average='micro')))
        print('Macro:')
        print("Recall: " + str(recall_score(test, predicted, average='macro')))
        print("Precision: " + str(precision_score(test, predicted, average='macro')))
        print("F1 Score: " + str(f1_score(test, predicted, average='macro')))
        print('Samples: ')
        print("Recall: " + str(recall_score(test, predicted, average='samples')))
        print("Precision: " + str(precision_score(test, predicted, average='samples')))
        print("F1 Score: " + str(f1_score(test, predicted, average='samples')))

    def results_labelled(self, Y_predict, Y_test):
        test = Y_test
        predicted = Y_predict
        labels = self.labels

        print('Weighted:')
        print("Recall: " + str(
            metrics.flat_recall_score(test, predicted, average='weighted', labels=labels, zero_division=0)))
        print("Precision: " + str(
            metrics.flat_precision_score(test, predicted, average='weighted', labels=labels, zero_division=0)))
        print("F1 Score: " + str(
            metrics.flat_f1_score(test, predicted, average='weighted', labels=labels, zero_division=0)))

        print("Recall: " + str(
            metrics.flat_recall_score(test, predicted, average='weighted', labels=labels, zero_division=1)))
        print("Precision: " + str(
            metrics.flat_precision_score(test, predicted, average='weighted', labels=labels, zero_division=1)))
        print("F1 Score: " + str(
            metrics.flat_f1_score(test, predicted, average='weighted', labels=labels, zero_division=1)))

        print('Micro:')
        print("Recall: " + str(
            metrics.flat_recall_score(test, predicted, average='micro', labels=labels, zero_division=0)))
        print("Precision: " + str(
            metrics.flat_precision_score(test, predicted, average='micro', labels=labels, zero_division=0)))
        print(
            "F1 Score: " + str(metrics.flat_f1_score(test, predicted, average='micro', labels=labels, zero_division=0)))

        print("Recall: " + str(
            metrics.flat_recall_score(test, predicted, average='micro', labels=labels, zero_division=1)))
        print("Precision: " + str(
            metrics.flat_precision_score(test, predicted, average='micro', labels=labels, zero_division=1)))
        print(
            "F1 Score: " + str(metrics.flat_f1_score(test, predicted, average='micro', labels=labels, zero_division=1)))

        print('Macro:')
        print("Recall: " + str(
            metrics.flat_recall_score(test, predicted, average='macro', labels=labels, zero_division=0)))
        print("Precision: " + str(
            metrics.flat_precision_score(test, predicted, average='macro', labels=labels, zero_division=0)))
        print(
            "F1 Score: " + str(metrics.flat_f1_score(test, predicted, average='macro', labels=labels, zero_division=0)))


def get_ortho_labels(orthographic: str):
    labels = []
    tmp = ''
    tag = False

    # Get all labels from orthographic form
    for char in orthographic:
        if char == '[':
            tag = True
        elif char == ']':
            labels.append(tmp)
            tag = False
            tmp = ''
        elif tag:
            tmp += char
    return labels


def get_surface_segments(surface: str):
    segments = []
    tmp = ''
    label = False

    # Get all segments from orthographic form
    for char in surface:
        if char == '[':
            segments.append(tmp)
            tmp = ''
            label = True
        elif char == ']':
            label = False
        elif not label:
            tmp += char
    return segments


def format_arrays_json(arr:[str], special_case = None):
    string = "["
    count = 0
    for seg in arr:
        if seg == special_case:
            string += "\"\\" + seg + "\""
        else:
            string += "\"" + seg + "\""
        if count != len(arr) - 1:
            string += ", "
        count += 1
    string += "]"
    return string

def get_surface_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('corpus_dir', type=str, help="the corpus directory")
    parser.add_argument('--corpus_dir', type=str, default='corpus_dir/', help="the corpus directory")
    #parser.add_argument('--model_dir', type=str, default="Neural CRF Model/surface_model", help="the output directory for model files")
    parser.add_argument('--model_dir', type=str, default="surface_model/",
                        help="the output directory for model files")

    parser.add_argument('--num_epoch', type=int, default=20, help="number of epoch to train")
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    #parser.add_argument('--weight_decay', type=float, default=0., help='the L2 normalization parameter')
    parser.add_argument('--weight_decay', type=float, default=0., help='the L2 normalization parameter')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size for training')
    parser.add_argument('--device', type=str, default=None,
                        help='the training device: "cuda:0", "cpu:0". It will be auto-detected by default')
    parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length within training')

    #parser.add_argument('--val_split', type=float, default=0.2, help='the split for the validation dataset')
    parser.add_argument('--val_split', type=float, default=0.2, help='the split for the validation dataset')
    #parser.add_argument('--test_split', type=float, default=0.2, help='the split for the testing dataset')
    parser.add_argument('--test_split', type=float, default=0.2, help='the split for the testing dataset')
    parser.add_argument('--recovery', action="store_true",
                        help="continue to train from the saved model in model_dir")
    parser.add_argument('--save_best_val_model', action="store_true",
                        help="save the model whose validation score is smallest")

    parser.add_argument('--embedding_dim', type=int, default=100, help='the dimension of the embedding layer')
    parser.add_argument('--hidden_dim', type=int, default=128, help='the dimension of the RNN hidden state')
    parser.add_argument('--num_rnn_layers', type=int, default=1, help='the number of RNN layers')
    parser.add_argument('--rnn_type', type=str, default="lstm", help='RNN type, choice: "lstm", "gru"')

    args = parser.parse_args()
    return args

def get_labelled_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('corpus_dir', type=str, help="the corpus directory")
    parser.add_argument('--corpus_dir', type=str, default='corpus_dir/', help="the corpus directory")

    parser.add_argument('--model_dir', type=str, default="labelled_model/", help="the output directory for model files")

    parser.add_argument('--num_epoch', type=int, default=20, help="number of epoch to train")
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    #parser.add_argument('--weight_decay', type=float, default=0., help='the L2 normalization parameter')
    parser.add_argument('--weight_decay', type=float, default=0., help='the L2 normalization parameter')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size for training')
    parser.add_argument('--device', type=str, default=None,
                        help='the training device: "cuda:0", "cpu:0". It will be auto-detected by default')
    parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length within training')

    #parser.add_argument('--val_split', type=float, default=0.2, help='the split for the validation dataset')
    parser.add_argument('--val_split', type=float, default=0.2, help='the split for the validation dataset')
    #parser.add_argument('--test_split', type=float, default=0.2, help='the split for the testing dataset')
    parser.add_argument('--test_split', type=float, default=0.2, help='the split for the testing dataset')
    parser.add_argument('--recovery', action="store_true",
                        help="continue to train from the saved model in model_dir")
    parser.add_argument('--save_best_val_model', action="store_true",
                        help="save the model whose validation score is smallest")

    parser.add_argument('--embedding_dim', type=int, default=100, help='the dimension of the embedding layer')
    parser.add_argument('--hidden_dim', type=int, default=128, help='the dimension of the RNN hidden state')
    parser.add_argument('--num_rnn_layers', type=int, default=1, help='the number of RNN layers')
    parser.add_argument('--rnn_type', type=str, default="lstm", help='RNN type, choice: "lstm", "gru"')

    args = parser.parse_args()
    return args


languages = ["zulu", "swati", "ndebele", "xhosa"]
for lang in languages:
    print("Language: " + lang)
    n = NeuralCRF(lang)
    x, y = n.surface_segmentation()
    n.results(x, y)
    print(lang + " Analysis complete.\n#############################################")

'''
n = NeuralCRF('zulu')
x, y = n.surface_segmentation()
n.results(x, y)
x, y = n.surface_labelled_segmentation()
n.results_labelled(x, y)
'''