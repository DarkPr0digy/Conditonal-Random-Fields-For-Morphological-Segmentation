from Symbols import Symbols
import Levenshtein as LevenshteinDistance


class FileReader:
    def __init__(self, filename):
        self.filename = filename
        file = open("morphology/" + filename, "r")
        self.lines = file.readlines()
        file = open("morphology/" + filename, "r")
        self.content = file.read()
        self.s = Symbols()

    def print(self):
        print(self.content)

    def reformat(self, filename):
        new_file = open("morphology/" + filename + ".conll", "w")
        first_line = True
        for line in self.lines:
            if first_line:
                first_line = not first_line
                continue

            is_blank = False
            if line == '' or line == '\n' or line == ' ':
                is_blank = True
            line_value = line.split("\t")

            try:
                is_int = isinstance(int(line_value[0]), int)
            except ValueError:
                is_int = False
            try:
                is_float = isinstance(float(line_value[0]), float)
            except ValueError:
                is_float = False

            if line_value[0] != "\n" and not self.s.inArr(line_value[0]) and \
                    not is_int and not is_float and not is_blank:
                new_file.write(removeLabels(line_value[0]) + " | " +
                               generateSurfaceSegmentation(removeLabels(line_value[0]),
                                                           removeLabels(line_value[3])) + " | " +
                               normaliseOrthographicForm(line_value[3]))
                print(removeLabels(line_value[0]) + " | " +
                      generateSurfaceSegmentation(removeLabels(line_value[0]), removeLabels(line_value[3])) + " | " +
                      normaliseOrthographicForm(line_value[3]))


def normaliseOrthographicForm(orthgraphic: str):
    str2_arr = []
    for char in orthgraphic:
        if char == '$' or char == '-':
            continue
        else:
            str2_arr.append(char)
    return "".join(str2_arr)


def removeLabels(str2: str):
    str2_arr = []
    last_seen_bracket = []
    # Can also use $
    for char in str2:
        if char == "(" or char == "[":
            # Do the thing
            last_seen_bracket.append(char)
            str2_arr.append("-")
        elif char == ")" or char == "]":
            # Do the other thing
            if len(last_seen_bracket) >= 1:
                last_seen_bracket.pop()
            else:
                continue
        elif char == "-" or char == '$':
            continue
        elif len(last_seen_bracket) >= 1:
            # Do the thing
            continue
        else:
            # Do the last thing
            str2_arr.append(char)

    if len(str2_arr) > 1:
        for i in range(len(str2_arr)):
            try:
                if str2_arr[i] == "-" and str2_arr[i - 1] == "-":
                    str2_arr.pop(i - 1)
                    # some segments have dual purpose, so this removes dual brackets
            except IndexError:
                continue

        if str2_arr[len(str2_arr) - 1] == "\n":
            str2_arr.pop()

        while str2_arr[len(str2_arr) - 1] == "-":
            str2_arr.pop()

    output = "".join(str2_arr)
    return output


def desegment(word: str):
    ans = ""
    for char in word:
        if char != "-":
            ans += char
    return ans


def generateSurfaceSegmentation(word: str, orthographic_form: str):
    if word.lower() == desegment(orthographic_form).lower():
        return orthographic_form
        # Lower because some words are capitalized and the de-segmented form of orthographic form
    else:
        replace = []
        edits = LevenshteinDistance.editops(desegment(orthographic_form), word)
        for x in edits:
            if x[0] == 'replace':
                replace.append(True)
            else:
                replace.append(False)
        if all(replace):
            # IF all operations are replace then I can just add dashes where they are in OG word
            dash_pos = [pos for pos in range(len(orthographic_form)) if
                        orthographic_form[pos] == '-']  # Get position of all dashes
            # dash_pos = [i for i, a in enumerate(orthographic_form) if a == "-"]
            print(dash_pos)
            arr = list(word)
            for d in dash_pos:
                arr.insert(d, "-")
            return "".join(arr)
        else:
            # GO through all operations
            segmented_form = list(orthographic_form)
            dash_pos = [pos for pos in range(len(orthographic_form)) if segmented_form[pos] == '-']
            desegmented_form = list(desegment(orthographic_form))

            for ops in edits:
                # print("Desegmented: " + str(desegmented_form) + "\nSegmented: " + str(segmented_form))
                if ops[0] == 'delete':
                    # THING
                    position = ops[2]
                    del desegmented_form[position]  # Remove from position in desegmented form
                    # print(position)
                    for x in dash_pos:
                        if position >= x:
                            # If there are dashes that occur before it
                            position += 1
                    # print(position)

                    try:
                        lone = segmented_form[position - 1] == '-' and segmented_form[position + 1] == '-'
                    except IndexError:
                        continue
                    del segmented_form[position]  # remove from position in segmented form
                    if lone:
                        del segmented_form[position]
                    dash_pos = [pos for pos in range(len(segmented_form)) if segmented_form[pos] == '-']
                elif ops[0] == 'replace':
                    # THING
                    # Nothing needs to be done to desegmented form because this has no net effect on length of word
                    position = ops[2]
                    # print(position)
                    # print(dash_pos)
                    for x in dash_pos:
                        if position >= x:
                            # If there are dashes that occur before it
                            position += 1
                    # print(position)
                    try:
                        if segmented_form[position - 1] == '-' and segmented_form[position + 1] == '-':
                            del segmented_form[position - 1]
                            # IF i need to updated dashes do it here
                            dash_pos = [pos for pos in range(len(segmented_form)) if segmented_form[pos] == '-']
                    except IndexError:
                        continue
            # print("Desegmented: " + str(desegmented_form) + "\nSegmented: " + str(segmented_form))
            dash_pos = [pos for pos in range(len(segmented_form)) if segmented_form[pos] == '-']
            surface_segmented = list(word)
            for dash in dash_pos:
                surface_segmented.insert(dash, '-')
            return "".join(surface_segmented).rstrip("-").lstrip("-")


languages = ["zulu", "swati", "ndebele", "xhosa"]

for lang in languages:
    inputFile = FileReader(lang + ".train.conll")
    inputFile.reformat(lang + ".train.clean")
    inputFile = FileReader(lang + ".test.conll")
    inputFile.reformat(lang + ".test.clean")
