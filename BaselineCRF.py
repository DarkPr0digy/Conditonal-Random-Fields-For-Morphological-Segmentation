import sklearn_crfsuite

training_data = {}
dev_data = {}
test_data = {}

lang="zulu"
samples = 0

input_files = ["morphology/" + lang+".train.clean.conll", "morphology/" + lang+".test.clean.conll"]
"""input_files = ["morphology/" + lang+".train.clean.conll",
"morphology/" + lang+".dev.clean.conll", "morphology/" + lang+".test.clean.conll"]"""

dictionaries = (training_data, test_data)
# dictionaries = (training_data, dev_data, test_data)


counter = 0
limit = 0
n_samples = samples # How many lines do I want to train it on
# If 0 do all?
for file in input_files:
    with open(file) as input_file:
        for row in input_file:
            line = row.split(" | ")
            result = []
            morph = ''
            tag = False

            # array with original word [0]
            # surface segmented word [1]
            # orthographic form[2] w. labels in square brackets

            for char in line[1]:
                # Surface Segmentation
                if char == '-':
                    if not morph:
                        result.append(morph)
                    morph = ''
                    continue
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
            dictionaries[counter][line[0]] = label
            limit += 1
        limit = 0
        counter += 1

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

def prepareData(word_dictionary, delta):
    X = []
    Y = []
    words = []

    for word in word_dictionary:
        wordStruct = '['+word+']'
        word_list = []
        word_label_list = []

        for i in range(len(wordStruct)):
            char_dict = {}
            for j in range(delta):
                char_dict['right_'+wordStruct[i:i + j + 1]] = 1
            for j in range(delta):
                char_dict['left_' + wordStruct[i - j - 1:i]] = 1

            char_dict['pos_start_'+str(i)] = 1
            """if wordStruct[i] in ['a','s','o']:
                char_dict[str(wordStruct[i])] = 1"""
            word_list.append(char_dict)
            if wordStruct[i] == '[':
                word_label_list.append('[')
            elif wordStruct[i] == ']':
                word_label_list.append(']')
            else:
                word_label_list.append(word_dictionary[word][i - 1])

        X.append(word_list)
        Y.append(word_label_list)
        temp_list_word = [char for char in wordStruct]
        words.append(temp_list_word)
    return X, Y, words

print("Features Computed")

best_epsilon, best_max_iteration, best_delta = 0, 0, 0
maxF1_score = 0

for epsilon in [0.001, 0.00001, 0.0000001]:
    for max_iterations in [80, 120, 160]:
        for delta in [3, 4, 5, 6, 7, 8, 9]:
            X_training, Y_training, words_training = prepareData(training_data, delta)
            X_test, Y_test,words_test = prepareData(training_data, delta)
            crf = sklearn_crfsuite.CRF(algorithm="ap",epsilon=epsilon, max_iterations=max_iterations)
            crf.fit(X_training, Y_training)
#################################################################


#################################################################
X_training, Y_training, words_training = prepareData(training_data, best_delta)
X_test, Y_test, words_test = prepareData(test_data, best_delta)
crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=best_epsilon, max_iterations=best_max_iteration)
crf.fit(X_training, Y_training)

Y_predict = crf.predict(X_test)
H, I, D = 0, 0, 0

for i in range(len(Y_test)):
    for j in range(len(Y_test[i])):
        if Y_test[j][i] == 'E' or Y_test[j][i] == 'S':
            if Y_test[j][i] == Y_predict[j][i]:
                H += 1
            else:
                D += 1
        else:
            if (Y_predict[j][i] == 'E' or Y_predict[j][i] == 'S'):
                I += 1
P = float(H)/(H+I)
R = float(H)/(H+D)
F1 = (2*P*R)/(P+R)
print('\nEvaluation on the Test set\n')
print('delta = ' + str(best_delta) + '\tNsamples = ' + str(n_samples) + '\tepsilon = ' + str(best_epsilon) + '\tmax_iter = ' + str(best_max_iteration))
print('Precision = ' + str(P))
print('Recall = ' + str(R))
print('F1-score = ' + str(F1))
print('\n' + str(n_samples) + '\t' + str(best_delta) + '\t' + str(best_epsilon) + '\t' + str(best_max_iteration) + '\t' + str(round(P, 3)) + '\t' + str(round(R, 3)) + '\t' + str(round(F1, 3)))




