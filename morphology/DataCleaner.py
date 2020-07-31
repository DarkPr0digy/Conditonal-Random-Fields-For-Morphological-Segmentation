from morphology.Symbols import Symbols
import Levenshtein as LevenshteinDistance
import os
import sys


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
                new_file.write(removeLabels(line_value[0]) + " | " +
                               generateSurfaceSegmentation(removeLabels(line_value[0]),
                                                           removeLabels(line_value[3])) + " | " +
                               normaliseOrthographicForm(line_value[3]))
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
    for char in orthographic:
        if char == '$' or char == '-':
            continue
        else:
            str2_arr.append(char)
    return "".join(str2_arr)


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

    return "".join(str2_arr).rstrip("-")


def de_segment(word: str):
    """Method used to de-segment the orthographic form of a word"""
    ans = ""
    for char in word:
        if char != "-":
            ans += char
    return ans


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

for lang in languages:
    print("Language: " + lang)
    # Add dev / validation dataset
    inputFile = DataCleaner(lang + ".train.conll")
    inputFile.reformat(lang + ".clean.train")
    inputFile = DataCleaner(lang + ".test.conll")
    inputFile.reformat(lang + ".clean.test")
    print(lang + " cleaning complete.\n#############################################")
