import os


def createFolderForFile(file, mode=0o775):
    path = os.path.dirname(file)
    os.makedirs(path, mode, True)
