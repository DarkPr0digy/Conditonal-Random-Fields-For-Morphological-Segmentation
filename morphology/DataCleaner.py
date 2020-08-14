from morphology.Symbols import Symbols
import Levenshtein as LevenshteinDistance
import os
import sys
import re


class DataCleaner:
    def __init__(self, filename: str):
        # Open file for reading only
        self.file = open(os.path.join(sys.path[0], filename), "r")
        self.lines = self.file.readlines()
        self.s = Symbols()

    def reformat(self, filename: str):
        """Method Used to Clean the Data contained in the original files, and to generate the surface segmentation"""
        new_file = open(os.path.join(sys.path[0], filename + ".conll"), "w")
        # open new file for writing
        first_line = True
        for line in self.lines:
            if first_line:
                first_line = not first_line
                continue
            # first line contains headings so skip that

            is_blank = False
            if line == '' or line == '\n' or line == ' ':
                is_blank = True
            # Check if line is whitespace
            line_value = line.split("\t")

            try:
                is_int = isinstance(int(line_value[0]), int)
            except ValueError:
                is_int = False
            # check if line is int
            try:
                is_float = isinstance(float(line_value[0]), float)
            except ValueError:
                is_float = False
            # Check if line is float

            if not self.s.inArr(line_value[0]) and not is_int and not is_float and not is_blank:
                # Formats as follows:
                # word | surface segmented form | orthographic segmented form
                orthographic_form = normaliseOrthographicForm(line_value[3].rstrip('\n'))
                surface_segmented = generateSurfaceSegmentation(removeLabels(line_value[0]),
                                                                                  removeLabels(line_value[3]))

                # print(line_value[0]+"|"+labelled_surface)
                new_file.write(removeLabels(line_value[0]) + " | " +
                               surface_segmented
                               + " | " + orthographic_form+'\n')
                """print(removeLabels(line_value[0]) + " | " +
                  generateSurfaceSegmentation(removeLabels(line_value[0]), removeLabels(line_value[3])) + " | " +
                  normaliseOrthographicForm(line_value[3]))"""

        # Close both files to avoid leakages
        self.file.close()
        new_file.close()


def normaliseOrthographicForm(orthographic: str):
    """Method to normalise the format of the orthographic form to make using it as input easier for the CRF"""
    # Formats orthographic form into following format
    # aa[bb]cc[dd]
    str2_arr = []

    # Removes Labels at the front of orthographic forms
    if orthographic[0] == '[':
        beginningLabel = True
        while beginningLabel:
            index = orthographic.find(']')
            orthographic = orthographic[index + 1: len(orthographic)]
            if orthographic[0] == '[':
                continue
            else:
                beginningLabel = False

    # Removes extra character
    for char in orthographic:
        if char == '$' or char == '-':
            continue
        else:
            str2_arr.append(char)

    # Combines double labels
    label = []
    str = []
    tag = False
    for i in range(len(str2_arr)):
        try:
            double_label = str2_arr[i] == ']' and str2_arr[i + 1] == '['
        except IndexError:
            double_label = False

        if str2_arr[i] == '[':
            tag = True
        elif double_label:
            label.append(',')
            label.append(' ')
        elif str2_arr[i] == ']':
            tag = False
            str.append('[')
            for char in label:
                str.append(char)
            str.append(']')
            label = []
        elif tag:
            label.append(str2_arr[i])
        else:
            str.append(str2_arr[i])

    return "".join(str)


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


def de_segment(word: str):
    """Method used to de-segment the orthographic form of a word"""
    ans = ""
    for char in word:
        if char != "-":
            ans += char
    return ans


def label_per_morpheme(orthographic: str):
    if orthographic.find('[') == -1:
        return False
    labels = []
    str = []
    orthographic += "|"
    is_label = False
    is_str = True
    tmp_label = ''
    tmp_str = ''
    for char in orthographic:
        if char == '[':
            if tmp_str:
                str.append(tmp_str)
            tmp_str = ''
            is_label = True
            is_str = False
        elif char == ']':
            if tmp_label:
                labels.append(tmp_label)
            tmp_label = ''
            is_label = False
            is_str = True
        elif char == "|":
            if is_str:
                if tmp_str:
                    str.append(tmp_str)
            elif is_label:
                if tmp_label:
                    labels.append(tmp_label)
        elif is_str:
            tmp_str += char
        elif is_label:
            tmp_label += char

    return len(labels) == len(str)


## Redundant
def generateLabelledSurfaceSegmentation(surface_segmented: str, orthographic_segmented: str):
    if removeLabels(orthographic_segmented) == surface_segmented:
        return orthographic_segmented
    else:
        surface = surface_segmented.split('-')
        ortho = removeLabels(orthographic_segmented).split('-')

        if len(surface) == len(ortho):
            # Segments are the same, labels can be directly translated
            labels = []
            tmp = ''
            tag = False

            # Get all labels from orthographic form
            for char in orthographic_segmented:
                if char == '[':
                    tag = True
                elif char == ']':
                    labels.append(tmp)
                    tag = False
                    tmp = ''
                elif tag:
                    tmp += char

            string = []

            for i in range(len(surface)):
                string.append(surface[i] + '[' + labels[i] + ']')
            return "".join(string)
        else:
            labels = []
            tmp = ''
            tag = False

            # Get all labels from orthographic form
            for char in orthographic_segmented:
                if char == '[':
                    tag = True
                elif char == ']':
                    labels.append(tmp)
                    tag = False
                    tmp = ''
                elif tag:
                    tmp += char

            edits = LevenshteinDistance.editops(removeLabels(orthographic_segmented), surface_segmented)

            ortho = list(removeLabels(orthographic_segmented))
            ortho_dashpos = [pos for pos in range(len(ortho)) if ortho[pos] == '-']

            """print(edits)
            print(ortho)
            print(surface_segmented)"""

            offset = 0
            for ops in edits:
                if ops[0] == 'delete':
                    # Delete Ops
                    position = ops[2] - offset
                    labelPos = 0
                    for x in ortho_dashpos:
                        if position >= x:
                            labelPos += 1

                    lone = False
                    try:
                        lone = ortho[position - 1] == '-' and ortho[position + 1] == '-'
                    except IndexError:
                        continue

                    """print(labels)
                    print(ortho_dashpos)
                    print(ortho)
                    print(position)
                    print(lone)"""

                    if lone:
                        del labels[labelPos]

                        del ortho[position]
                        del ortho[position]
                        ortho_dashpos = [pos for pos in range(len(ortho)) if ortho[pos] == '-']
                        offset += 1
                    else:
                        try:
                            del ortho[position]
                        except IndexError:
                            print(surface_segmented)
                            print(orthographic_segmented)
                            print(edits)
                            print(labels)
                            print(surface)
                            print(position)
                            print(position + offset)
                            print(ortho)
                            quit(0)
                        ortho_dashpos = [pos for pos in range(len(ortho)) if ortho[pos] == '-']
                elif ops[0] == 'insert':
                    position = ops[2] - offset
                    char = surface_segmented[ops[1]]
                    ortho.insert(position, char)
                    ortho_dashpos = [pos for pos in range(len(ortho)) if ortho[pos] == '-']

            str = []
            for x in range(len(surface)):
                try:
                    str.append(surface[x] + '[' + labels[x] + ']')
                except IndexError:
                    print(labels)
                    print(surface)
                    print(orthographic_segmented)
                    print(ortho)
                    quit(0)
            return "".join(str)


def generateSurfaceSegmentation(word: str, orthographic_form: str):
    """Method used to generate the surface segmentation of a word
    given the word and the orthographic form of the word"""

    if word.lower() == de_segment(orthographic_form).lower():
        # If the word and the orthographic form are the same,
        # then the orthographic form is the surface form and this can be returned
        return orthographic_form
    else:
        replace = []
        # Generate list of operations needed to turn de-segmented orthographic form into original word
        # of the form [(operation, source pos, destination pos)...]
        edits = LevenshteinDistance.editops(de_segment(orthographic_form), word)

        for x in edits:
            if x[0] == 'replace':
                replace.append(True)
            else:
                replace.append(False)
        if all(replace):
            # If all operations being performed on a word are replace operations then one can simply add dashes to
            # the word where they appear in the orthographic form to generate the surface form

            # Get position of all dashes in the orthographic form
            dash_pos = [pos for pos in range(len(orthographic_form)) if orthographic_form[pos] == '-']
            arr = list(word)
            for d in dash_pos:
                arr.insert(d, "-")
            return "".join(arr)

        else:
            segmented_form = list(orthographic_form)
            # Get position of all dashes in the segmented form
            dash_pos = [pos for pos in range(len(segmented_form)) if segmented_form[pos] == '-']
            de_segmented_form = list(de_segment(orthographic_form))

            for ops in edits:
                # Iterate through operations one by one and try to reverse them
                if ops[0] == 'delete':
                    # Get position of the change
                    position = ops[2]

                    # Remove from position in de-segmented form
                    del de_segmented_form[position]

                    # Position of same char in segmented form will be higher if there are dashes that occur before it
                    # This loop accounts for that difference
                    for x in dash_pos:
                        if position >= x:
                            position += 1

                    try:
                        # try determine if the segment is an lone character such as
                        # ...-x-...
                        lone = segmented_form[position - 1] == '-' and segmented_form[position + 1] == '-'
                    except IndexError:
                        continue

                    # Remove from position in segmented form
                    del segmented_form[position]

                    if lone:
                        # if it is a lone character, also delete the preceding dash
                        del segmented_form[position]

                    # Update positions of the dashes
                    dash_pos = [pos for pos in range(len(segmented_form)) if segmented_form[pos] == '-']
                elif ops[0] == 'replace':
                    # Nothing needs to be done to de-segmented form because
                    # this has no net effect on construction of the word

                    # Get position of the change
                    position = ops[2]

                    # Position of same char in segmented form will be higher if there are dashes that occur before it
                    # This loop accounts for that difference
                    for x in dash_pos:
                        if position >= x:
                            position += 1
                    try:
                        if segmented_form[position - 1] == '-' and segmented_form[position + 1] == '-':
                            # if the changed character is a lone character delete preceding dash
                            del segmented_form[position - 1]

                            # Update position of dashes
                            dash_pos = [pos for pos in range(len(segmented_form)) if segmented_form[pos] == '-']
                    except IndexError:
                        continue

            # Final update of position of dashes
            dash_pos = [pos for pos in range(len(segmented_form)) if segmented_form[pos] == '-']
            surface_segmented = list(word)
            for dash in dash_pos:
                # where there is a dash in updated segmented form put a dash in the word
                surface_segmented.insert(dash, '-')

            return "".join(surface_segmented).rstrip("-").lstrip("-")


languages = ["zulu", "swati", "ndebele", "xhosa"]

# print(generateLabelledSurfaceSegmentation('wa-s-e-Ningizimu', 'wa[PossConc1]s[PreLoc-s]e[LocPre]i[NPrePre5]li[BPre5]Ningizimu[NStem]'))

# print(generateSurfaceSegmentation('SOHLELO',removeLabels('sa[PossConc7]u[NPrePre11]lu[BPre11]hlelo[NStem]'), 'sa[PossConc7]u[NPrePre11]lu[BPre11]hlelo[NStem]'))

for lang in languages:
    print("Language: " + lang)
    inputFile = DataCleaner(lang + ".train.conll")
    inputFile.reformat(lang + ".clean.train")
    inputFile = DataCleaner(lang + ".test.conll")
    inputFile.reformat(lang + ".clean.test")
    print(lang + " cleaning complete.\n#############################################")
