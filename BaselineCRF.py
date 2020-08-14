import sklearn_crfsuite
import time


class BaselineCRF:
    def __init__(self, language: str):
        self.language = language
        self.input_files = ["morphology/" + self.language + ".clean.train.conll",
                            "morphology/" + self.language + ".clean.dev.conll",
                            "morphology/" + self.language + ".clean.test.conll"]

    def surface_segmentation(self):
        """This method is used to perform the surface segmentation"""
        tic = time.perf_counter()
        # Collect the Data
        ##################################################
        training_data, dev_data, test_data = {}, {}, {}
        dictionaries = (training_data, dev_data, test_data)
        counter = 0
        for file in self.input_files:
            input_file = open(file, 'r')
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
        print("Data Collected in " + str(tic - toc.__round__(4)) + " seconds")

        # Compute Features & Optimise Model Using Dev Set
        ##################################################
        best_epsilon, best_max_iteration, best_delta = 0, 0, 0
        maxF1 = 0
        print("Beginning Feature Computation and Model Optimisation")
        tic = time.perf_counter()

        """for epsilon in [0.001, 0.00001, 0.0000001]:
            for max_iterations in [80, 120, 160]:
                for delta in [1]:  # 3, 4, 5, 6, 7, 8, 9
                    X_training, Y_training, words_training = surface_segment_data_preparation(training_data, delta)
                    X_dev, Y_dev, words_dev = surface_segment_data_preparation(dev_data, delta)
                    # X_test, Y_test, words_test = surface_segment_data_preparation(test_data, delta)

                    # crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=epsilon, max_iterations=max_iterations)

                    crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=epsilon, max_iterations=max_iterations)
                    #crf = sklearn_crfsuite.CRF(algorithm=algo, max_iterations=max_iterations)
                    # crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)
                    crf.fit(X_training, Y_training)

                    Y_predict = crf.predict(X_dev)

                    # print("Epsilon: " + str(epsilon) + "\nMax Iterations: " + str(max_iterations) + "\nDelta: " + str(delta))

                    true_positives, false_positives, false_negatives = 0, 0, 0
                    for j in range(len(Y_dev)):
                        for i in range(len(Y_dev[j])):
                            if Y_dev[j][i] == 'E' or Y_dev[j][i] == 'S':
                                if Y_dev[j][i] == Y_predict[j][i]:
                                    true_positives += 1
                                else:
                                    false_negatives += 1
                            else:
                                if Y_predict[j][i] == 'E' or Y_predict[j][i] == 'S':
                                    false_positives += 1
                    try:
                        precision = float(true_positives) / (true_positives + false_positives)
                        recall = float(true_positives) / (true_positives + false_negatives)
                        f1 = (2 * precision * recall) / (precision + recall)
                        if maxF1 < f1:
                            maxF1 = f1
                            best_epsilon = epsilon
                            best_max_iteration = max_iterations
                            best_delta = delta
                    except ZeroDivisionError:
                        continue"""

        toc = time.perf_counter()
        print("Features Successfully Computed & Model Optimised " + str(tic - toc.__round__(4) // 60) + " seconds")

        # Evaluate Model On the Test Set Using Optimised Model
        #######################################################

        best_delta = 8
        best_epsilon = 0.0000001
        best_max_iteration = 80
        best_algo = 'ap'

        a, b, c = surface_segment_data_preparation(training_data, best_delta)
        print("X_Training: " + str(a[len(a) - 1]) + "\n################################")
        print("Y_training: " + str(b[len(b) - 1]) + "\n################################")
        print("Words Training: " + str(c[len(c) - 1]) + "\n############################")

        X_training, Y_training, words_training = surface_segment_data_preparation(training_data, best_delta)
        X_dev, Y_dev, words_dev = surface_segment_data_preparation(dev_data, best_delta)
        X_test, Y_test, words_test = surface_segment_data_preparation(test_data, best_delta)
        crf = sklearn_crfsuite.CRF(algorithm=best_algo, epsilon=best_epsilon, max_iterations=best_max_iteration)
        crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)

        Y_predict = crf.predict(X_test)

        print(Y_predict[0])


        true_positives, false_positives, false_negatives = 0, 0, 0
        for j in range(len(Y_test)):
            for i in range(len(Y_test[j])):
                if Y_test[j][i] == 'E' or Y_test[j][i] == 'S' or Y_test[j][i] == 'B':
                    if Y_test[j][i] == Y_predict[j][i]:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if Y_predict[j][i] == 'E' or Y_predict[j][i] == 'S' or Y_predict[j][i] == 'B':
                        false_positives += 1
                """if Y_test[j][i] == 'E' or Y_test[j][i] == 'S':
                    if Y_test[j][i] == Y_predict[j][i]:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if Y_predict[j][i] == 'E' or Y_predict[j][i] == 'S':
                        false_positives += 1"""
        precision = float(true_positives) / (true_positives + false_positives)
        recall = float(true_positives) / (true_positives + false_negatives)
        f1 = (2 * precision * recall) / (precision + recall)
        print('\nEvaluation on the Test set\n')
        print('delta = ' + str(best_delta) + '\tepsilon = ' + str(
            best_epsilon) + '\tmax_iter = ' + str(best_max_iteration) + '\tBest Algo = ' + best_algo)
        print('Precision = ' + str(precision))
        print('Recall = ' + str(recall))
        print('F1-score = ' + str(f1))
        print(str(round(precision, 3)) + '\t' + str(round(recall, 3)) + '\t' + str(round(f1, 3)))

        ############################################################################################

        crf = sklearn_crfsuite.CRF(algorithm=best_algo, epsilon=best_epsilon, max_iterations=best_max_iteration,
                                   all_possible_transitions=True,all_possible_states=True)
        crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)

        Y_predict = crf.predict(X_test)

        print(Y_predict[0])

        true_positives, false_positives, false_negatives = 0, 0, 0
        for j in range(len(Y_test)):
            for i in range(len(Y_test[j])):
                if Y_test[j][i] == 'E' or Y_test[j][i] == 'S':
                    if Y_test[j][i] == Y_predict[j][i]:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if Y_predict[j][i] == 'E' or Y_predict[j][i] == 'S':
                        false_positives += 1
        precision = float(true_positives) / (true_positives + false_positives)
        recall = float(true_positives) / (true_positives + false_negatives)
        f1 = (2 * precision * recall) / (precision + recall)
        print('\nEvaluation on the Test set\n')
        print('delta = ' + str(best_delta) + '\tepsilon = ' + str(
            best_epsilon) + '\tmax_iter = ' + str(best_max_iteration) + '\tBest Algo = ' + best_algo)
        print('Precision = ' + str(precision))
        print('Recall = ' + str(recall))
        print('F1-score = ' + str(f1))
        print(str(round(precision, 3)) + '\t' + str(round(recall, 3)) + '\t' + str(round(f1, 3)))

        ############################################################################################

        crf = sklearn_crfsuite.CRF(algorithm=best_algo, epsilon=best_epsilon, max_iterations=best_max_iteration,
                                   all_possible_states=True)
        crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)

        Y_predict = crf.predict(X_test)

        print(Y_predict[0])

        true_positives, false_positives, false_negatives = 0, 0, 0
        for j in range(len(Y_test)):
            for i in range(len(Y_test[j])):
                if Y_test[j][i] == 'E' or Y_test[j][i] == 'S':
                    if Y_test[j][i] == Y_predict[j][i]:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if Y_predict[j][i] == 'E' or Y_predict[j][i] == 'S':
                        false_positives += 1
        precision = float(true_positives) / (true_positives + false_positives)
        recall = float(true_positives) / (true_positives + false_negatives)
        f1 = (2 * precision * recall) / (precision + recall)
        print('\nEvaluation on the Test set\n')
        print('delta = ' + str(best_delta) + '\tepsilon = ' + str(
            best_epsilon) + '\tmax_iter = ' + str(best_max_iteration) + '\tBest Algo = ' + best_algo)
        print('Precision = ' + str(precision))
        print('Recall = ' + str(recall))
        print('F1-score = ' + str(f1))
        print(str(round(precision, 3)) + '\t' + str(round(recall, 3)) + '\t' + str(round(f1, 3)))

        ############################################################################################

        crf = sklearn_crfsuite.CRF(algorithm=best_algo, epsilon=best_epsilon, max_iterations=best_max_iteration,
                                   all_possible_transitions=True)
        crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)

        Y_predict = crf.predict(X_test)

        print(Y_predict[0])

        true_positives, false_positives, false_negatives = 0, 0, 0
        for j in range(len(Y_test)):
            for i in range(len(Y_test[j])):
                if Y_test[j][i] == 'E' or Y_test[j][i] == 'S':
                    if Y_test[j][i] == Y_predict[j][i]:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if Y_predict[j][i] == 'E' or Y_predict[j][i] == 'S':
                        false_positives += 1
        precision = float(true_positives) / (true_positives + false_positives)
        recall = float(true_positives) / (true_positives + false_negatives)
        f1 = (2 * precision * recall) / (precision + recall)
        print('\nEvaluation on the Test set\n')
        print('delta = ' + str(best_delta) + '\tepsilon = ' + str(
            best_epsilon) + '\tmax_iter = ' + str(best_max_iteration) + '\tBest Algo = ' + best_algo)
        print('Precision = ' + str(precision))
        print('Recall = ' + str(recall))
        print('F1-score = ' + str(f1))
        print(str(round(precision, 3)) + '\t' + str(round(recall, 3)) + '\t' + str(round(f1, 3)))

        ############################################################################################
        crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=best_epsilon)
        crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)

        Y_predict = crf.predict(X_test)

        print("No Max Iterations")

        true_positives, false_positives, false_negatives = 0, 0, 0
        for j in range(len(Y_test)):
            for i in range(len(Y_test[j])):
                if Y_test[j][i] == 'E' or Y_test[j][i] == 'S':
                    if Y_test[j][i] == Y_predict[j][i]:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if Y_predict[j][i] == 'E' or Y_predict[j][i] == 'S':
                        false_positives += 1
        precision = float(true_positives) / (true_positives + false_positives)
        recall = float(true_positives) / (true_positives + false_negatives)
        f1 = (2 * precision * recall) / (precision + recall)
        print('\nEvaluation on the Test set\n')
        print('delta = ' + str(best_delta) + '\tepsilon = ' + str(
            best_epsilon))
        print('Precision = ' + str(precision))
        print('Recall = ' + str(recall))
        print('F1-score = ' + str(f1))
        print(str(round(precision, 3)) + '\t' + str(round(recall, 3)) + '\t' + str(round(f1, 3)))

        ################################################################################

        crf = sklearn_crfsuite.CRF(algorithm='ap', max_iterations=best_max_iteration)
        crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)

        Y_predict = crf.predict(X_test)

        print("No Epsilon")

        true_positives, false_positives, false_negatives = 0, 0, 0
        for j in range(len(Y_test)):
            for i in range(len(Y_test[j])):
                if Y_test[j][i] == 'E' or Y_test[j][i] == 'S':
                    if Y_test[j][i] == Y_predict[j][i]:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if Y_predict[j][i] == 'E' or Y_predict[j][i] == 'S':
                        false_positives += 1
        precision = float(true_positives) / (true_positives + false_positives)
        recall = float(true_positives) / (true_positives + false_negatives)
        f1 = (2 * precision * recall) / (precision + recall)
        print('\nEvaluation on the Test set\n')
        print('delta = ' + str(best_delta) + '\tmax_iter = ' + str(best_max_iteration))
        print('Precision = ' + str(precision))
        print('Recall = ' + str(recall))
        print('F1-score = ' + str(f1))
        print(str(round(precision, 3)) + '\t' + str(round(recall, 3)) + '\t' + str(round(f1, 3)))

        #########################################################

        crf = sklearn_crfsuite.CRF(algorithm='lbfgs', epsilon=best_epsilon, max_iterations=best_max_iteration)
        crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)

        Y_predict = crf.predict(X_test)

        print("LBFGS Algo")

        true_positives, false_positives, false_negatives = 0, 0, 0
        for j in range(len(Y_test)):
            for i in range(len(Y_test[j])):
                if Y_test[j][i] == 'E' or Y_test[j][i] == 'S':
                    if Y_test[j][i] == Y_predict[j][i]:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if Y_predict[j][i] == 'E' or Y_predict[j][i] == 'S':
                        false_positives += 1
        precision = float(true_positives) / (true_positives + false_positives)
        recall = float(true_positives) / (true_positives + false_negatives)
        f1 = (2 * precision * recall) / (precision + recall)
        print('\nEvaluation on the Test set\n')
        print('delta = ' + str(best_delta) + '\tepsilon = ' + str(
            best_epsilon) + '\tmax_iter = ' + str(best_max_iteration))
        print('Precision = ' + str(precision))
        print('Recall = ' + str(recall))
        print('F1-score = ' + str(f1))
        print(str(round(precision, 3)) + '\t' + str(round(recall, 3)) + '\t' + str(round(f1, 3)))

        ####################################################################

        crf = sklearn_crfsuite.CRF(algorithm='pa', epsilon=best_epsilon, max_iterations=best_max_iteration)
        crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)

        Y_predict = crf.predict(X_test)

        print("pa")

        true_positives, false_positives, false_negatives = 0, 0, 0
        for j in range(len(Y_test)):
            for i in range(len(Y_test[j])):
                if Y_test[j][i] == 'E' or Y_test[j][i] == 'S':
                    if Y_test[j][i] == Y_predict[j][i]:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if Y_predict[j][i] == 'E' or Y_predict[j][i] == 'S':
                        false_positives += 1
        precision = float(true_positives) / (true_positives + false_positives)
        recall = float(true_positives) / (true_positives + false_negatives)
        f1 = (2 * precision * recall) / (precision + recall)
        print('\nEvaluation on the Test set\n')
        print('delta = ' + str(best_delta) + '\tepsilon = ' + str(
            best_epsilon) + '\tmax_iter = ' + str(best_max_iteration))
        print('Precision = ' + str(precision))
        print('Recall = ' + str(recall))
        print('F1-score = ' + str(f1))
        print(str(round(precision, 3)) + '\t' + str(round(recall, 3)) + '\t' + str(round(f1, 3)))

        #########################################################################################

        crf = sklearn_crfsuite.CRF(algorithm='arow', epsilon=best_epsilon, max_iterations=best_max_iteration)
        crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)

        Y_predict = crf.predict(X_test)

        print("arow")

        true_positives, false_positives, false_negatives = 0, 0, 0
        for j in range(len(Y_test)):
            for i in range(len(Y_test[j])):
                if Y_test[j][i] == 'E' or Y_test[j][i] == 'S':
                    if Y_test[j][i] == Y_predict[j][i]:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if Y_predict[j][i] == 'E' or Y_predict[j][i] == 'S':
                        false_positives += 1
        precision = float(true_positives) / (true_positives + false_positives)
        recall = float(true_positives) / (true_positives + false_negatives)
        f1 = (2 * precision * recall) / (precision + recall)
        print('\nEvaluation on the Test set\n')
        print('delta = ' + str(best_delta) + '\tepsilon = ' + str(
            best_epsilon) + '\tmax_iter = ' + str(best_max_iteration))
        print('Precision = ' + str(precision))
        print('Recall = ' + str(recall))
        print('F1-score = ' + str(f1))
        print(str(round(precision, 3)) + '\t' + str(round(recall, 3)) + '\t' + str(round(f1, 3)))
        ############################################################################################

    def orthographic_labelled_segmentation(self):
        """This Method is used to perform the orthographic labelled segmentation"""
        # Collect the Data
        ##################################################
        tic = time.perf_counter()
        training_data, dev_data, test_data = {}, {}, {}
        dictionaries = (training_data, dev_data, test_data)
        counter = 0

        for file in self.input_files:
            input_file = open(file, 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")

                # Just Orthographic form
                ##############################################
                dictionaries[counter][content[0]] = [content[1], removeLabels(content[2])]
                ##############################################
            input_file.close()
            counter += 1

        toc = time.perf_counter()
        print("Data Collected in " + str(tic - toc.__round__(4)) + " seconds")

        # Run the CRF
        ################################################################
        best_delta = 8
        best_epsilon = 0.0000001
        best_max_iteration = 80
        best_algo = 'ap'

        X_training, Y_training, words_training = labelled_orthographic_data_preparation(training_data)
        X_dev, Y_dev, words_dev = labelled_orthographic_data_preparation(dev_data)
        X_test, Y_test, words_test = labelled_orthographic_data_preparation(test_data)
        crf = sklearn_crfsuite.CRF(algorithm=best_algo, epsilon=best_epsilon, max_iterations=best_max_iteration)
        crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)

        Y_predict = crf.predict(X_test)

        print(Y_predict[0])

        H, I, D = 0, 0, 0
        for j in range(len(Y_test)):
            for i in range(len(Y_test[j])):
                if Y_test[j][i] == 'E' or Y_test[j][i] == 'S':
                    if Y_test[j][i] == Y_predict[j][i]:
                        H += 1
                    else:
                        D += 1
                else:
                    if Y_predict[j][i] == 'E' or Y_predict[j][i] == 'S':
                        I += 1
        precision = float(H) / (H + I)
        recall = float(H) / (H + D)
        f1 = (2 * precision * recall) / (precision + recall)
        print('\nEvaluation on the Test set\n')
        print('delta = ' + str(best_delta) + '\tepsilon = ' + str(
            best_epsilon) + '\tmax_iter = ' + str(best_max_iteration) + '\tBest Algo = ' + best_algo)
        print('Precision = ' + str(precision))
        print('Recall = ' + str(recall))
        print('F1-score = ' + str(f1))
        print(str(round(precision, 3)) + '\t' + str(round(recall, 3)) + '\t' + str(round(f1, 3)))

def surface_segment_data_preparation(word_dictionary: {str, str}, delta: int):
    """"This Method is used to prepare data for the crf that is performing the surface segmentation"""
    """
    X = []
    Y = []
    words = []

    for word in word_dictionary:
        word_list = []
        word_label_list = []

        for i in range(len(word)):
            char_dict = {}
            for j in range(delta):
                char_dict['right_' + word[i:i + j + 1]] = 1
            for j in range(delta):
                char_dict['left_' + word[i - j - 1:i]] = 1

            char_dict['pos_start_' + str(i)] = 1

            word_list.append(char_dict)

            word_label_list.append(word_dictionary[word][i])

        X.append(word_list)
        Y.append(word_label_list)
        temp_list_word = [char for char in word]
        words.append(temp_list_word)
    return X, Y, words"""

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
            gram_dict[word[i]] = 1
            #gram_dict[word[i]] = word_dictionary[word][i]
            gram_arr.append(word[i])

            ### BIGRAM
            try:
                tmp = word[i - 1: i + 1]
                if tmp:
                    gram_dict[tmp] = 1
                    #gram_dict[tmp] = word_dictionary[word][i-1:i+1]
                    gram_arr.append(tmp)
            except IndexError:
                continue
            try:
                tmp = word[i: i + 2]
                if tmp:
                    gram_dict[tmp] = 1
                    #gram_dict[tmp] = word_dictionary[word][i:i + 2]
                    gram_arr.append(tmp)
            except IndexError:
                continue

            ### TRIGRAM
            try:
                tmp = word[i - 1: i + 2]
                if tmp:
                    gram_dict[tmp] = 1
                    #gram_dict[tmp] = word_dictionary[word][i - 1:i + 2]
                    gram_arr.append(tmp)
            except IndexError:
                continue

            ##  FourGram
            try:
                tmp = word[i - 1: i + 3]
                if tmp:
                    gram_dict[tmp] = 1
                    #gram_dict[tmp] = word_dictionary[word][i - 1:i + 3]
                    gram_arr.append(tmp)
            except IndexError:
                continue

            try:
                tmp = word[i - 2: i + 2]
                if tmp:
                    gram_dict[tmp] = 1
                    #gram_dict[tmp] = word_dictionary[word][i - 2:i + 2]
                    gram_arr.append(tmp)
            except IndexError:
                continue

            ## FiveGram
            try:
                tmp = word[i - 2: i + 3]
                if tmp:
                    gram_dict[tmp] = 1
                    #gram_dict[tmp] = word_dictionary[word][i - 2:i + 3]
                    gram_arr.append(tmp)
            except IndexError:
                continue

            ## SixGram
            try:
                tmp = word[i - 3: i + 3]
                if tmp:
                    gram_dict[tmp] = 1
                    # gram_dict[tmp] = word_dictionary[word][i - 2:i + 3]
                    gram_arr.append(tmp)
            except IndexError:
                continue

            try:
                tmp = word[i - 2: i + 4]
                if tmp:
                    gram_dict[tmp] = 1
                    # gram_dict[tmp] = word_dictionary[word][i - 2:i + 3]
                    gram_arr.append(tmp)
            except IndexError:
                continue

            ## SevenGram
            try:
                tmp = word[i - 3: i + 4]
                if tmp:
                    gram_dict[tmp] = 1
                    # gram_dict[tmp] = word_dictionary[word][i - 2:i + 3]
                    gram_arr.append(tmp)
            except IndexError:
                continue

            ## EightGram
            try:
                tmp = word[i - 4: i + 4]
                if tmp:
                    gram_dict[tmp] = 1
                    # gram_dict[tmp] = word_dictionary[word][i - 2:i + 3]
                    gram_arr.append(tmp)
            except IndexError:
                continue

            try:
                tmp = word[i - 3: i + 5]
                if tmp:
                    gram_dict[tmp] = 1
                    # gram_dict[tmp] = word_dictionary[word][i - 2:i + 3]
                    gram_arr.append(tmp)
            except IndexError:
                continue

            word_list.append(gram_dict)
            word_label_list.append(word_dictionary[word][i])

        X.append(word_list)
        Y.append(word_label_list)
        words.append([char for char in word])
    return X, Y, words

def labelled_orthographic_data_preparation(word_dictionary: {str, str}):
    """This Method is used to prepare data for the crf that is performing the labelled orthographic segmentation"""
    X = []
    Y = []
    words = []


    for word in word_dictionary:
        #########################################
        # Just Orthographic Form
        # Words
        surface = word_dictionary[word][0].split('-')
        orthographic = word_dictionary[word][1].split('-')

        while len(surface) < len(orthographic):
            surface.append('')

        words.append(char for char in word)
        # Surface Form
        X.append(surface)
        # orthographic form
        Y.append(orthographic)
        #########################################

    return X, Y, words


"""
        #Orthographic Form
        for char in line[2]:
                        if char == '-':
                            if not morph == '':
                                result.append(morph)
                            morph = ''
                            continue
                        elif char == '[' and not tag:
                            if not morph == '':
                                result.append(morph)
                            morph = ''
                            tag = True
                        elif not tag:
                            morph += char
                        elif tag and char == ']':
                            tag = False
                        else:
                            continue

                    if morph.strip():
                        result.append(morph.strip())
                    # print(result)

                    label = ''
                    for morph in result:
                        if len(morph) == 1:
                            label += 'S'
                        else:
                            label += 'B'
                            for i in range(len(morph) - 2):
                                label += 'M'
                            label += 'E'
                    dictionaries[counter][line[0]] = label
                    #print(dictionaries)
                    limit += 1
                    #if limit > n_samples:
                        #break
                limit = 0
                counter += 1"""

"""languages = ["zulu", "swati", "ndebele", "xhosa"]
for lang in languages:
    print("Language: "+lang)
    crf = BaselineCRF(lang)
    crf.surface_segmentation()
    print("####################")"""

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

crf = BaselineCRF("zulu")
crf.surface_segmentation()
# crf.orthographic_labelled_segmentation()
