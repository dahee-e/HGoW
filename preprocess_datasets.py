import os
from shutil import copyfile
import re
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
def text_clean (text):
    data = []
    text = re.sub('\n', '', text)
    text = re.sub('-', ' ', text)
    text = text.lower()
    doc = nlp(text)
    for token in doc:
        if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_ != 'PRON') and (token.pos_ != 'SPACE'):
            data.append(token.lemma_)
    return data

def copy_files(input_file, output_file):
    copyfile(input_file, output_file)
def preprocess_data(input_file, output_file):
    # Process aacy file
    # result = ""
    # with open(input_file, 'r') as f:
    #     for line in f:
    #         tokens = line.split()
    #         temp = []
    #         for token in tokens:
    #             token = token.split("/")[0]
    #             temp.append(token)
    #         token_result = " ".join(temp)
    #         result += token_result + "\n"
    #
    # # Write the result to output file
    # with open(output_file, 'w') as f:
    #     f.write(result)
    #
    result = []
    with open(input_file, 'r') as f:
        for line in f:
            result.append(text_clean (line))
    result = sum(result, [])
    result = ",".join(set(result))

    # Write the result to output file
    with open(output_file, 'w') as f:
        f.write(result)

# Example usage

parent_folder = "/Users/dahee/PycharmProjects/pythonProject9/datasets/Hulth2003/validation_training/validation/"
parent_folder = "/Users/dahee/PycharmProjects/pythonProject9/datasets/Marujo2012/train_test/train/"
file = "/Users/dahee/PycharmProjects/pythonProject9/datasets/semeval/"
file_path = "test.combined.stem.final"
# Traverse through the subfolders
# for file_path in os.listdir(parent_folder):
#     file = os.path.join(parent_folder, file_path)
#     # if file.endswith('.stc'):
#     #     preprocess_data(file,parent_folder+file_path.split('_')[0]+".stc")
#     # if file.endswith('.key'):
#     #     preprocess_data(file,parent_folder+file_path.split('.')[0]+".pre")
#     #     print(file_path.split('.')[0]+".pre")
#     # if file.endswith('.pre'):
#     #     copy_files(file,parent_folder+file_path.split('.')[0]+".key")
#     #     print(file_path.split('.')[0]+".key")
#     if file.endswith('.final'):
#         copy_files(file,parent_folder+file_path.split('.')[0]+".key")
#         print(file_path.split('.')[0]+".key")

def preprocess_data_semeval(file_path, file_name):


    with open(file_path+file_name, 'r') as f:
        for line in f:
            result = []
            output_file = line.split(' : ')[0]
            result.append(text_clean (line.split(' : ')[1]))
            result = sum(result, [])
            result = ",".join(set(result))
    # Write the result to output file
            with open(file_path+output_file+".key", 'w') as f:
                f.write(result)

if __name__ == '__main__':
    parent_folder = "/Users/dahee/PycharmProjects/pythonProject9/datasets/Hulth2003/validation_training/validation/"
    parent_folder = "/Users/dahee/PycharmProjects/pythonProject9/datasets/Marujo2012/train_test/train/"
    file = "/Users/dahee/PycharmProjects/pythonProject9/datasets/semeval/"
    file_path = "test.combined.stem.final"
    if file_path.endswith('.final'):
        preprocess_data_semeval(file,file_path)

