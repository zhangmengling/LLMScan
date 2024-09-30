'''
Author: zhangmengling zhangmengdi1997@126.com
Date: 2024-07-16 23:19:20
LastEditors: zhangmengling zhangmengdi1997@126.com
LastEditTime: 2024-08-11 23:47:55
FilePath: /mengdizhang/LLM_probing_Causality/experiments_alpaca_vicuna/generate_alpaca_vicuna_logprobs.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
import os
sys.path.append(os.getcwd())

import argparse
import os

import dotenv
import numpy as np
import openai
import pandas as pd
import pickle


from lllm.llama_utils import establish_llama_endpoint
from lllm.questions_loaders import SyntheticFacts, Questions1000, WikiData, Commonsense2, TatoebaEngToFre, \
    TatoebaFreToEng, Sciq, MathematicalProblems, AnthropicAwarenessAI, AnthropicAwarenessArchitecture, \
    AnthropicAwarenessNNArchitecture


def find_number_of_rows_with_questions_the_model_can_answer(number_answerable_questions_required, dataset, model):
    return np.where(dataset[f"{model}_can_answer"].cumsum() == number_answerable_questions_required)[0][0] + 1


dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

parser = argparse.ArgumentParser()
# parser.add_argument("--model", type=str, default='alpaca')
# parser.add_argument("--dataset", type=str, default='questions1000')
# parser.add_argument("--number_questions_to_answer", "-n", type=int, default=160)
# args = parser.parse_args()
# print(f'Testing {args.model} on {args.dataset}')
# model = args.model
# number_questions_to_answer = args.number_questions_to_answer

# '''
# model_path = "meta-llama/"
# model_name = "Llama-2-13b-chat-hf"

# ------------------------------------------- Generating x_train -------------------------------------------
# '''
# model_path = "meta-llama/"
# model_name = "Meta-Llama-3.1-8B-Instruct"
# model_name = "Llama-2-13b-chat-hf"

model_path = "mistralai/"
model_name = "Mistral-7B-Instruct-v0.2"

model = model_name

llama_endpoint = establish_llama_endpoint(model)
print(f'{model} endpoint established.')

datasets = [Questions1000(), WikiData(), Commonsense2(), Sciq(), MathematicalProblems()]
datasets = [Questions1000()]

dataset = Questions1000()
print("-->dataset.columns", list(dataset.columns))

for dataset in datasets:
    print("-->dataset_name", dataset.__class__.__name__)
    print("-->dataset:", dataset.columns)
    dataset.generate_logprobs(
        max_questions_to_try=None,
        model_suspect=model,
        model_kwargs_suspect={"endpoint": llama_endpoint, "max_tokens": 64, "stop": "\n"},
        save_progress=False,
        append_results_to_existing_ones=False  # False: overwrite them
    )

print("GENERATE_ALPACA_VICUNA_LOGPROBS COMPLETED CORRECTLY")
# '''

# ------------------------------------------- Detecting -------------------------------------------
'''
model_name = "Llama-2-13b-chat-hf"
model_name = "Meta-Llama-3.1-8B-Instruct"
# model_name = "Mistral-7B-Instruct-v0.2"

datasets = [Questions1000()]
dataset_names = [dataset.__class__.__name__ for dataset in datasets]

def extract_logprobs(model):
    logprobs_dataset = []
    # probs_dataset = []

    for d, name in zip(datasets, dataset_names):
        df = d.get_logprobs_differences(model_suspect=model, return_setup=True)
        # df2 = d.get_probs_differences(return_setup=True)
        df["dataset"] = [name] * len(df)
        logprobs_dataset.append(df)
        # probs_dataset.append(df2)

    logprobs_dataset = pd.concat(logprobs_dataset)
    # probs_dataset = pd.concat(probs_dataset)
    return logprobs_dataset

logprobs_dataset_llama13b = extract_logprobs(model_name)

print(logprobs_dataset_llama13b.shape)

logprobs_dataset_llama13b.iloc[:, 0] = logprobs_dataset_llama13b.iloc[:, 0].apply(lambda x: np.array(x))
logprobs_dataset_llama13b.iloc[:, 1] = logprobs_dataset_llama13b.iloc[:, 1].apply(lambda x: np.array(x))

# extract the logprobs and put them in a array
X_llama13b = np.concatenate([np.stack([logprobs_dataset_llama13b.iloc[i, 0] for i in range(logprobs_dataset_llama13b.shape[0])]),
                          np.stack([logprobs_dataset_llama13b.iloc[i, 1] for i in range(logprobs_dataset_llama13b.shape[0])])])
# create y
y_llama13b = np.concatenate([np.ones(len(logprobs_dataset_llama13b)), np.zeros(len(logprobs_dataset_llama13b))])


# classifiers for subsets:
with open("./results/trained_classifiers/logistic_logprobs_classifier_no_lie.pkl", "rb") as f:
    classifier_no_lie = pickle.load(f)
with open("./results/trained_classifiers/logistic_logprobs_classifier_lie.pkl", "rb") as f:
    classifier_lie = pickle.load(f)
with open("./results/trained_classifiers/logistic_logprobs_classifier_knowable.pkl", "rb") as f:
    classifier_knowable = pickle.load(f)
with open("./results/trained_classifiers/logistic_logprobs_classifier_all_probes.pkl", "rb") as f:
    classifier_all = pickle.load(f)
with open("./results/trained_classifiers/logistic_logprobs_classifier_subsets_union.pkl", "rb") as f:
    classifier_subsets_union = pickle.load(f)

classifiers_logprobs_list = [classifier_no_lie, classifier_lie, classifier_knowable, classifier_all, classifier_subsets_union]

# same for binary instead of logprobs:
with open("./results/trained_classifiers/logistic_binary_classifier_no_lie.pkl", "rb") as f:
    classifier_no_lie_binary = pickle.load(f)
with open("./results/trained_classifiers/logistic_binary_classifier_lie.pkl", "rb") as f:
    classifier_lie_binary = pickle.load(f)
with open("./results/trained_classifiers/logistic_binary_classifier_knowable.pkl", "rb") as f:
    classifier_knowable_binary = pickle.load(f)
with open("./results/trained_classifiers/logistic_binary_classifier_all_probes.pkl", "rb") as f:
    classifier_all_binary = pickle.load(f)
with open("./results/trained_classifiers/logistic_binary_classifier_subsets_union.pkl", "rb") as f:
    classifier_subsets_union_binary = pickle.load(f)

classifiers_binary_list = [classifier_no_lie_binary, classifier_lie_binary, classifier_knowable_binary, 
                           classifier_all_binary, classifier_subsets_union_binary]

all_indices = np.arange(X_llama13b.shape[1])
no_lie_indices = np.load("./results/probes_groups/no_lie_indices.npy")
lie_indices = np.load("./results/probes_groups/lie_indices.npy")
knowable_indices = np.load("./results/probes_groups/knowable_indices.npy")
subsets_union_indices = np.concatenate([no_lie_indices, lie_indices, knowable_indices])
indeces_list = [no_lie_indices, lie_indices, knowable_indices, all_indices, subsets_union_indices]

name_list = ["no_lie", "lie", "knowable", "all", "subsets_union"]


classification_results_df = pd.DataFrame(
    columns=["model", "accuracy", "auc", "y_pred", "y_pred_proba", "labels", "binary", "subset"])
models = [model_name]

for binary in [False, True]:

    classifier_list = classifiers_binary_list if binary else classifiers_logprobs_list

    for model_name, X, y in zip(models, [X_llama13b], [y_llama13b]):

        for classifier, suffix, index_list in zip(classifier_list, name_list, indeces_list):
            
            X_item = X[:, index_list]

            if binary:
                X_item = np.array(X_item) > 0

            print("-->classifier", classifier, type(classifier))
            accuracy, auc, _, y_pred, y_pred_proba = classifier.evaluate(X_item, y, return_ys=True)

            classification_results_df = pd.concat([classification_results_df, pd.DataFrame(
                {"model": [model_name], "accuracy": [accuracy], "auc": [auc], "y_pred": [y_pred],
                 "y_pred_proba": [y_pred_proba], "labels": [y], "binary": [binary], "subset": [suffix]})])

print(classification_results_df[["model", "binary", "subset", "accuracy", "auc"]])

'''