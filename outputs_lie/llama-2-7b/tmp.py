'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-04-18 19:50:29
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-04-18 19:56:11
FilePath: /mengdizhang/LLM-LieDetector/outputs/llama-2-7b/tmp.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
def save_to_json(saving_dir, file_name, dataset_name, aie_orig, kurt_orig, aie_lie, kurt_lie):
    with open(saving_dir + file_name, 'r') as file:
        data = json.load(file)

    index = 1
    new_dataset_name = dataset_name
    while new_dataset_name in data:
        index += 1
        new_dataset_name = f"{dataset_name}_{index}"

    data[new_dataset_name] = {
        "aie_orig": aie_orig,
        "aie_lie": aie_lie,
        "kurt_orig": kurt_orig,
        "kurt_lie": kurt_lie
    }

    with open(saving_dir + file_name, 'w') as file:
        json.dump(data, file)


saving_dir = "./"
file_name = "AIE_lie_test.json"
dataset_name = "TEST"
aie_orig = [0, 0, 0, 0, 0]
kurt_orig = 10
aie_lie = [0, 0, 0, 0, 0]
kurt_lie = 10
save_to_json(saving_dir, file_name, dataset_name, aie_orig, kurt_orig, aie_lie, kurt_lie)