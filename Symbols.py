class Symbols:
    def __init__(self):
        self.arr = []
        self.arr.append([",", "_", "PUNCT", "_"])
        self.arr.append([".", "_", "PUNCT", "_"])
        self.arr.append(["[", "_", "PUNCT", "_"])
        self.arr.append(["]", "_", "PUNCT", "_"])
        self.arr.append(["(", "_", "PUNCT", "_"])
        self.arr.append([")", "_", "PUNCT", "_"])
        self.arr.append(["/", "_", "PUNCT", "_"])
        self.arr.append([":", "_", "PUNCT", "_"])
        self.arr.append(["©", "_", "PUNCT", "_"])
        self.arr.append([";", "_", "PUNCT", "_"])
        self.arr.append(["\'", "_", "PUNCT", "_"])
        self.arr.append(["\"", "_", "PUNCT", "_"])
        self.arr.append(["", "_", "PUNCT", "_"])
        self.arr.append([" ", "_", "PUNCT", "_"])
        self.arr.append(["-", "_", "PUNCT", "_"])
        self.arr.append(["!", "_", "PUNCT", "_"])
        self.arr.append(["?", "_", "PUNCT", "_"])

    def inArr(self, symbol):
        for symbols in self.arr:
            if symbols[0] == symbol:
                return True
        return False
