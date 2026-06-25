from os import path, listdir
from decimal import Decimal
import shutil

# returns a string with num rounded to 3 decimal places
# num is intended to be a float
def float_to_rounded_string(num):
    return str(round(Decimal(num), 3))

# returns a list of the file names in directory
# this is not a full path, just the file name
def get_files_in_directory(directory):
    return [f for f in listdir(directory) if path.isfile(path.join(directory, f))]

# inputs an array of file paths and copies those files to the specified directory
def copy_files_in_directory(files, output_directory):
    for file in files:
        shutil.copy(file, output_directory)    


# returns all the folder names in directory
# this is not a full path, just the subdirectory name
def get_folders_in_directory(directory):
   return [f for f in listdir(directory) if not path.isfile(path.join(directory, f))]

# saves the openpyxl Workbook object "wb" in location output_directory/output_file_name
# if output_file_name does not end in .xlsx, it is added before saving
def output_sheet(output_directory, output_file_name, wb):
    if output_file_name[-5:] != ".xlsx":
        output_file_name = output_file_name + ".xlsx"
    wb.save(path.join(output_directory, output_file_name))

    