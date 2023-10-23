import glob
from documents.folder import Folder


class Library:
    """
    The library stores and manages files.
    """

    def __init__(self, chunking_strategy):
        self.folders = {}
        self.chunking_strategy = chunking_strategy

    def create_folder(self, name: str):
        if name not in self.folders:
            self.folders[name] = Folder(self.chunking_strategy)
        else:
            raise KeyError

    def load_folder(self, name: str, path: str):
        self.create_folder(name)
        files = glob.glob(f"{path}/*")
        for file in files:
            self.folders[name].add_document(file)

    def __str__(self) -> str:
        return str(self.folders.values())
