import numpy as np
def evaluation_pre(data, answer_path):
    with open(answer_path, 'r') as f:
        answer = f.read()
        answer = answer.split(",")
    total = list(set(answer+data))
    binary_answer = np.zeros(len(total))
    binary_data = np.zeros(len(total))
    for i in range(len(total)):
        if total[i] in answer:
            binary_answer[i] = 1
        if total[i] in data:
            binary_data[i] = 1
    return binary_data, binary_answer