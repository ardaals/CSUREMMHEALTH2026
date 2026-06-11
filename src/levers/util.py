from os import path, listdir

# returns a list of the file names in directory
# this is not a full path, just the file name
def get_files_in_directory(directory):
    return [f for f in listdir(directory) if path.isfile(path.join(directory, f))]

# returns all the folder names in directory
# this is not a full path, just the subdirectory name
def get_folders_in_directory(directory):
   return [f for f in listdir(directory) if not path.isfile(path.join(directory, f))]
