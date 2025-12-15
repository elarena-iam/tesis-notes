from DatasetReader import DatasetReader

class DevDataset:

    def __init__(self, path):
        datasetReader = DatasetReader(1000, path)
        self.data = datasetReader.load()
