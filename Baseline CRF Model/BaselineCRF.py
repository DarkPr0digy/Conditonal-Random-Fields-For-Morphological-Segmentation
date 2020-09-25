import sklearn_crfsuite
import sklearn
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn_crfsuite import metrics
from sklearn.preprocessing import MultiLabelBinarizer
import time
import os
import sys


class BaselineCRF:
    def __init__(self, language: str):
        self.input_files = ["../morphology/" + language + '/' + language + ".clean.train.conll",
                            "../morphology/" + language + '/' + language + ".clean.dev.conll",
                            "../morphology/" + language + '/' + language + ".clean.test.conll"]

    def surface_segmentation(self):
        """This method is used to perform the surface segmentation"""
        tic = time.perf_counter()
        # Collect the Data
        ##################################################
        training_data, dev_data, test_data = {}, {}, {}
        dictionaries = (training_data, dev_data, test_data)
        counter = 0
        for file in self.input_files:
            input_file = open(os.path.join(sys.path[0], file), 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")
                result = []
                morph = ''
                tag = False

                for char in content[1]:
                    # Surface Segmentation
                    if char == '-':
                        result.append(morph)
                        morph = ''
                    else:
                        morph += char

                if morph.strip():
                    result.append(morph.strip())

                label = ''
                for morph in result:
                    if len(morph) == 1:
                        label += 'S'
                    else:
                        label += 'B'
                        for i in range(len(morph) - 2):
                            label += 'M'
                        label += 'E'
                # current dictionary being referenced
                # Key is word and value is segmented form
                # print(content)
                dictionaries[counter][content[0]] = label

            input_file.close()
            counter += 1

        toc = time.perf_counter()
        print("Data Collected in " + str(tic - toc.__round__(2)))

        # Compute Features & Optimise Model Using Dev Set
        ##################################################
        best_epsilon, best_max_iteration = 0, 0
        maxF1 = 0
        print("Beginning Feature Computation and Model Optimisation")
        tic = time.perf_counter()

        '''for epsilon in [0.001, 0.00001, 0.0000001]:
            for max_iterations in [80, 120, 160]:
                X_training, Y_training, words_training = surface_segment_data_preparation(training_data)
                X_dev, Y_dev, words_dev = surface_segment_data_preparation(dev_data)
                crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=epsilon, max_iterations=max_iterations)
                crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)

                Y_predict = crf.predict(X_dev)
                Y_dev = MultiLabelBinarizer().fit_transform(Y_dev)
                Y_predict = MultiLabelBinarizer().fit_transform(Y_predict)
                f1 = f1_score(Y_dev, Y_predict, average='micro')
                if f1 > maxF1:
                    f1 = maxF1
                    best_epsilon = epsilon
                    best_max_iteration = max_iterations

        print(best_max_iteration)
        print(best_epsilon)'''

        toc = time.perf_counter()
        print("Features Successfully Computed & Model Optimised " + str(tic - toc.__round__(2)))

        # Evaluate Model On the Test Set Using Optimised Model
        #######################################################

        best_max_iteration = 160
        best_epsilon = 1e-07

        #a, b, c = surface_segment_data_preparation(training_data)
        #print("X_Training: " + str(a[len(a) - 1]) + "\n################################")
        #print("Y_training: " + str(b[len(b) - 1]) + "\n################################")
        #print("Words Training: " + str(c[len(c) - 1]) + "\n############################")

        X_training, Y_training, words_training = surface_segment_data_preparation(training_data)
        X_dev, Y_dev, words_dev = surface_segment_data_preparation(dev_data)
        X_test, Y_test, words_test = surface_segment_data_preparation(test_data)
        crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=best_epsilon, max_iterations=best_max_iteration)
        crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)

        Y_predict = crf.predict(X_test)
        return Y_predict, Y_test

    def surface_labelled_segmentation(self):
        """This method is used to label the surface segments"""
        tic = time.perf_counter()

        # Collect the data
        ###########################################
        training_data, dev_data, test_data = {}, {}, {}
        dictionaries = (training_data, dev_data, test_data)
        counter = 0
        for file in self.input_files:
            input_file = open(os.path.join(sys.path[0], file), 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")
                labels = '-'.join(get_labels(content[2]))
                segments = removeLabels(content[2])

                # dictionaries[counter][content[0]] = [segments, labels] # word:[[segments],[labels]]
                dictionaries[counter][segments] = labels  # segments : labels
            input_file.close()
            counter += 1

        toc = time.perf_counter()
        print("Data Collected in " + str(tic - toc.__round__(2)))

        # Evaluate Model On the Test Set Using Optimised Model
        #######################################################


        best_delta = 8
        best_epsilon = 0.0000001
        best_max_iteration = 160
        best_algo = 'ap'

        best_epsilon, best_max_iteration = 0, 0
        maxF1 = 0
        print("Beginning Feature Computation and Model Optimisation")
        tic = time.perf_counter()

        '''for epsilon in [0.001, 0.00001, 0.0000001]:
            for max_iterations in [80, 120, 160, 200]:
                X_training, Y_training, words_training = surface_labelled_data_preparation(training_data)
                X_dev, Y_dev, words_dev = surface_labelled_data_preparation(dev_data)
                crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=epsilon, max_iterations=max_iterations)
                crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)

                Y_predict = crf.predict(X_dev)
                # f1 = f1_score(Y_dev, Y_predict, average='micro')
                labels = list(crf.classes_)
                sorted_labels = sorted(labels)
                f1 = metrics.flat_f1_score(Y_dev, Y_predict, average='micro', labels=labels, zero_division=0)
                if f1 > maxF1:
                    f1 = maxF1
                    best_epsilon = epsilon
                    best_max_iteration = max_iterations

        print(best_max_iteration)
        print(best_epsilon)'''

        X_training, Y_training, words_training = surface_labelled_data_preparation(training_data)
        X_dev, Y_dev, words_dev = surface_labelled_data_preparation(dev_data)
        X_test, Y_test, words_test = surface_labelled_data_preparation(test_data)
        print("Data Processed")

        best_epsilon = 1e-07
        best_max_iteration = 280
        best_algo = 'ap'

        # crf = sklearn_crfsuite.CRF(algorithm=best_algo, epsilon=best_epsilon, max_iterations=best_max_iteration)
        '''crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )'''
        crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=best_epsilon, max_iterations=best_max_iteration)
        print("CRF Initialized")
        #crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)
        crf.fit(X_training, Y_training)
        print("Data Fitted")
        Y_predict = crf.predict(X_test)
        #print(Y_predict[0])
        #print(Y_test[0])
        labels = list(crf.classes_)
        sorted_labels = sorted(labels)
        return Y_predict, Y_test, sorted_labels

    def results(self, Y_predict, Y_test):
        # print(metrics.flat_accuracy_score(Y_test, Y_predict))
        # print(metrics.flat_precision_score(Y_test, Y_predict))
        # print(metrics.flat_f1_score(Y_test, Y_predict))

        test = MultiLabelBinarizer().fit_transform(Y_test)
        predicted = MultiLabelBinarizer().fit_transform(Y_predict)

        print('Micro:')
        print("Recall: " + str(recall_score(test, predicted, average='micro')))
        print("Precision: " + str(precision_score(test, predicted, average='micro')))
        print("F1 Score: " + str(f1_score(test, predicted, average='micro')))

    def results_labelled(self, Y_predict, Y_test, labels):

        test = Y_test
        predicted = Y_predict
        print('Micro:')
        print("Recall: " + str(metrics.flat_recall_score(test, predicted, average='micro', labels=labels, zero_division=0)))
        print("Precision: " + str(metrics.flat_precision_score(test, predicted, average='micro', labels=labels, zero_division=0)))
        print("F1 Score: " + str(metrics.flat_f1_score(test, predicted, average='micro', labels=labels, zero_division=0)))

def surface_segment_data_preparation(word_dictionary: {str, str}):
    """"This Method is used to prepare data for the crf that is performing the surface segmentation"""
    X = []
    Y = []
    words = []
    for word in word_dictionary:
        word_list = []
        word_label_list = []
        for i in range(len(word)):
            gram_dict = {}
            gram_arr = []

            ### Unigram
            # gram_dict[word[i]] = 1
            gram_dict["uni_" + word[i]] = 1
            gram_arr.append(word[i])

            ### BIGRAM
            try:
                tmp = word[i - 1: i + 1]
                if tmp:
                    # gram_dict[tmp] = 1
                    gram_dict["bi_" + tmp] = 1
                    gram_arr.append(tmp)
            except IndexError:
                continue
            try:
                tmp = word[i: i + 2]
                if tmp:
                    # gram_dict[tmp] = 1
                    gram_dict["bi_" + tmp] = 1
                    gram_arr.append(tmp)
            except IndexError:
                continue

            ### TRIGRAM
            try:
                tmp = word[i - 1: i + 2]
                if tmp:
                    # gram_dict[tmp] = 1
                    gram_dict["tri_" + tmp] = 1
                    gram_arr.append(tmp)
            except IndexError:
                continue

            ##  FourGram
            try:
                tmp = word[i - 1: i + 3]
                if tmp:
                    # gram_dict[tmp] = 1
                    gram_dict["four_" + tmp] = 1
                    gram_arr.append(tmp)
            except IndexError:
                continue

            try:
                tmp = word[i - 2: i + 2]
                if tmp:
                    # gram_dict[tmp] = 1
                    gram_dict["four_" + tmp] = 1
                    gram_arr.append(tmp)
            except IndexError:
                continue

            ## FiveGram
            try:
                tmp = word[i - 2: i + 3]
                if tmp:
                    # gram_dict[tmp] = 1
                    gram_dict["five_" + tmp] = 1
                    gram_arr.append(tmp)
            except IndexError:
                continue

            ## SixGram
            try:
                tmp = word[i - 3: i + 3]
                if tmp:
                    # gram_dict[tmp] = 1
                    gram_dict["six_" + tmp] = 1
                    gram_arr.append(tmp)
            except IndexError:
                continue

            try:
                tmp = word[i - 2: i + 4]
                if tmp:
                    # gram_dict[tmp] = 1
                    gram_dict["six_" + tmp] = 1
                    gram_arr.append(tmp)
            except IndexError:
                continue

            if word[i] in 'aeiou':
                gram_dict["vowel"] = 1
            else:
                gram_dict["const"] = 1

            if word[i].isupper():
                gram_dict["upper"] = 1
            else:
                gram_dict["lower"] = 1

            word_list.append(gram_dict)
            word_label_list.append(word_dictionary[word][i])

        X.append(word_list)
        Y.append(word_label_list)
        words.append([char for char in word])
    return X, Y, words

def surface_labelled_data_preparation(word_dictionary: {str, str}):
    # nge-zin-konzo : NPre-BPre-NStem
    X = []
    Y = []
    words = []

    for word in word_dictionary:
        segments = word.split('-')
        labels = word_dictionary[word].split('-')
        segment_features = []
        for i in range(len(segments)):
            features = {}

            segment_length = len(segments[i])
            features['length'] = segment_length

            features['segment.lower()'] = segments[i].lower()
            features['pos_in_word'] = i

            if segment_length % 2 == 0:
                features['even'] = 1
            else:
                features['odd'] = 1

            features['begin'] = segments[i][0]
            features['end'] = segments[i][len(segments[i]) - 1]

            try:
                features['prev_segment'] = segments[i-1]
            except IndexError:
                features['prev_segment'] = ''
                #continue

            try:
                features['next_segment'] = segments[i + 1]
            except IndexError:
                features['next_segment'] = ''

            if segments[0].isupper():
                features['start_upper'] = 1
            else:
                features['start_lower'] = 1

            if segments[0] in 'aeiou':
                features['first_vowel'] = 1
            else:
                features['first_const'] = 1

            segment_features.append(features)
        words.append(segments)

        X.append(segment_features)
        Y.append(labels)
        words.append(word)

    return X, Y, words

def removeLabels(str2: str):
    """Method to remove labels from the orthographic segmentation so this form
    can be used to generate the surface segmentation"""
    str2_arr = []
    last_seen_bracket = []
    for char in str2:
        if char == "(" or char == "[":
            last_seen_bracket.append(char)
            str2_arr.append("-")
        elif char == ")" or char == "]":
            if len(last_seen_bracket) >= 1:
                last_seen_bracket.pop()
            else:
                continue
        elif char == "-" or char == '$':
            continue
        elif len(last_seen_bracket) >= 1:
            continue
        else:
            str2_arr.append(char)

    if len(str2_arr) > 1:
        for i in range(len(str2_arr)):
            try:
                if str2_arr[i] == "-" and str2_arr[i - 1] == "-":
                    str2_arr.pop(i - 1)
                    # Some segments have dual purpose, so this removes dual dashes that result from this
            except IndexError:
                continue

        if str2_arr[len(str2_arr) - 1] == "\n":
            str2_arr.pop()

    return "".join(str2_arr).rstrip("-").lstrip("-")


def get_labels(orthographic: str):
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

def x_run_average_surface(num:int, language:str):
    recall, precision, f1 = [], [], []
    for i in range(num):
        CRF = BaselineCRF(language)
        x, y = CRF.surface_segmentation()

        test = MultiLabelBinarizer().fit_transform(y)
        predicted = MultiLabelBinarizer().fit_transform(x)

        recall.append(recall_score(test, predicted, average='micro'))
        precision.append(precision_score(test, predicted, average='micro'))
        f1.append(f1_score(test, predicted, average='micro'))

    recall = sum(recall) / len(recall)
    precision = sum(precision) / len(precision)
    f1 = sum(f1) / len(f1)
    return recall, precision, f1

def x_run_average_labelled(num:int, language:str):
    recall, precision, f1 = [], [], []
    for i in range(num):
        CRF = BaselineCRF(lang)
        predict, test, labels = CRF.surface_labelled_segmentation()

        recall.append(metrics.flat_recall_score(test, predict, average='micro', labels=labels, zero_division=0))
        precision.append(metrics.flat_precision_score(test, predict, average='micro', labels=labels, zero_division=0))
        f1.append(metrics.flat_f1_score(test, predict, average='micro', labels=labels, zero_division=0))

    recall = sum(recall) / len(recall)
    precision = sum(precision) / len(precision)
    f1 = sum(f1) / len(f1)
    return recall, precision, f1


languages = ["zulu", "swati", "ndebele", "xhosa"]

for lang in languages:
    print("Language: " + lang)
    x, y, z = x_run_average_labelled(5, lang)
    print("recall: "+str(x))
    print("precision: "+str(y))
    print("f1: "+str(z))
    #CRF = BaselineCRF(lang)
    #x, y = CRF.surface_segmentation()
    #CRF.results(x, y)
    #x, y, z = CRF.surface_labelled_segmentation()
    #CRF.results_labelled(x, y, z)
    print(lang + " cleaning complete.\n#############################################")
