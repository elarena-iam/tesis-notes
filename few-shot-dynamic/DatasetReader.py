import json

class DatasetReader:

    def __init__(self, register_count, path):
        self.register_count = register_count
        self.path = path

    def load(self):
        f = open(self.path, "r")
        data = []
        lines = f.readlines()
        if (self.register_count == 0):
            for line in lines:
                data.append(json.loads(line))
        else:
            for i in range(self.register_count):
                data.append(json.loads(lines[i]))
        f.close()
        return data
