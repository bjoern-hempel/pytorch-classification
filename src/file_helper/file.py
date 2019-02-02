import os
import glob
import platform
import re


def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime


def create_folder_for_file(file, mode=0o775):
    path = os.path.dirname(file)
    os.makedirs(path, mode, True)


def search_files(path, name):
    files = []
    for file in glob.iglob('{}/**/{}'.format(path, name), recursive=True):
        files.append(file)

    return files

def analyse_file_and_get_config(file):

    # build file template
    file_template = re.sub(
        r"/(csv|models)/",
        "/{}/",
        re.sub(
            r"/(settings|summary_full|summary|validated|checkpoint|model_best)_",
            "/{}_",
            re.sub(
                r"\.(csv)$",
                ".{}",
                file
            )
        )
    )

    # settings of all files to create
    settings = {
        'csv.settings': ['csv', 'settings', 'csv'],
        'csv.summary_full': ['csv', 'summary_full', 'csv'],
        'csv.summary': ['csv', 'summary', 'csv'],
        'csv.validated': ['csv', 'validated', 'csv'],

        'model.checkpoint': ['models', 'checkpoint', 'pth'],
        'model.model_best': ['models', 'model_best', 'pth'],

        'pdf.confusion_matrix_val': ['charts', 'confusion_matrix_val', 'pdf'],
        'png.confusion_matrix_val': ['charts', 'confusion_matrix_val', 'png'],

        'pdf.mds': ['charts', 'mds', 'pdf'],
        'png.mds': ['charts', 'mds', 'png']
    }

    config = {
        'files': {},
        'property_path': os.path.dirname(file.split('/processed/')[1]).replace('/elements/', '/')
    }

    for key, values in settings.items():
        file_path = file_template.format(values[0], values[1], values[2])

        config['files'][key] = {
            'path': file_path,
            'exists': os.path.isfile(file_path)
        }

    return config