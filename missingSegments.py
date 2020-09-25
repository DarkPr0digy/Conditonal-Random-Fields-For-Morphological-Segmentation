import sys
import os


class missingSegments:
    def __init__(self, language:str):
        self.language = language
        self.input_files = ["morphology/" + language + '/' + language + ".clean.train.conll",
                            "morphology/" + language + '/' + language + ".clean.test.conll"]
        self.test_labels = []
        self.test_segments = []
        self.train_labels = []
        self.train_segments = []

    def getSegments(self):
        counter = 0
        for file in self.input_files:
            input_file = open(os.path.join(sys.path[0], file), 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")
                surface_form = content[1].split('-')
                label_list = get_ortho_labels(content[2])

                if counter == 0:
                    for segment in surface_form:
                        if segment not in self.train_segments:
                            self.train_segments.append(segment)

                    for label in label_list:
                        if label not in self.train_labels:
                            self.train_labels.append(label)
                else:
                    for segment in surface_form:
                        if segment not in self.test_segments:
                            self.test_segments.append(segment)

                    for label in label_list:
                        if label not in self.test_labels:
                            self.test_labels.append(label)
            counter += 1

    def getSegmentsDictionary(self):
        train_dict = {}
        test_dict = {}
        counter = 0
        for file in self.input_files:
            input_file = open(os.path.join(sys.path[0], file), 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")
                surface_form = content[1].split('-')
                label_list = get_ortho_labels(content[2])

                if counter == 0:
                    for i in range(len(surface_form)):
                        if surface_form[i] in train_dict:
                            if label_list[i] not in train_dict[surface_form[i]]:
                                train_dict[surface_form[i]].append(label_list[i])
                        else:
                            train_dict[surface_form[i]] = [label_list[i]]
                else:
                    for i in range(len(surface_form)):
                        if surface_form[i] in test_dict:
                            if label_list[i] not in test_dict[surface_form[i]]:
                                test_dict[surface_form[i]].append(label_list[i])
                        else:
                            test_dict[surface_form[i]] = [label_list[i]]
            counter += 1
        return train_dict, test_dict



    def missing_test_segments(self):
        missing_segments = []
        #print(sorted(self.train_segments))
        #print(sorted(self.test_segments))
        for segment in self.test_segments:
            if segment not in self.train_segments:
                missing_segments.append(segment)
        return missing_segments

    def missing_test_labels(self):
        missing_labels = []
        #print(sorted(self.train_labels))
        #print(sorted(self.test_labels))
        for label in self.test_labels:
            if label not in self.train_labels:
                missing_labels.append(label)
        return missing_labels

    def labels_and_segments(self):
        both = []
        for label in self.train_labels:
            if label in self.train_segments:
                both.append(label)
        for label in self.test_labels:
            if label in self.test_segments:
                both.append(label)
        return both




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


languages = ["zulu", "swati", "ndebele", "xhosa"]
for lang in languages:
    print("Language: " + lang)
    m = missingSegments(lang)
    #m.getSegments()
    x, y = m.getSegmentsDictionary()
    for seg in x.keys():
        if len(x[seg]) >= 2:
            print(seg + ": " + str(x[seg]))
    print("##############################################################")
    for seg in y.keys():
        if len(x[seg]) >= 2:
            print(seg + ": " + str(y[seg]))
    print("Labels and Segments: "+str(m.labels_and_segments()))
    #print("Missing Segments: "+str(m.missing_test_segments()))
    #print("Missing Labels: " + str(m.missing_test_labels()))
    print(lang + " Analysis complete.\n#############################################")
