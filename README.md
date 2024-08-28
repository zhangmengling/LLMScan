# LLM Lie Detection based on Causality Analysis. 
This repository is to detect LLM's lie behavior based on causality analysis. 
This repository contains code from paper "[How to catch an AI liar: Lie detection in black-box LLMs by asking unrelated questions](https://arxiv.org/abs/2309.15840)" with [GitHub link](https://github.com/LoryPack/LLM-LieDetector.git) and paper "[CASPER: Causality Analysis for Evaluating the Security of Large Language Models](https://arxiv.org/abs/2312.07876)" with [GitHub link](https://github.com/casperllm/CASPER.git) for refernece. 

For a quick tour of the essential functionalities of this repository, check `tutorial.ipynb`.

## Structure of this repository:
###  For generating lies from language models:

- `data` contains the Q/A datasets which are used to generate lies. These are a set of 10 existing dataset plus one which we introduce. `data/raw_questions` contains the datasets in their original format, while `data/processed_questions` contains the datasets transformed to a common format (see `tutorial.ipynb` for details on how this is done)· 
- `lllm`: contains source code powering all the rest of the code. In particular, `questions_loader.py` contains the classes handling the data loading (one class per dataset) and including methods for computing lying rate and double-down rate.
- `lying_rate_double_down_rate_probes.ipynb` prompts GPT-3.5 to lie, evaluates its lying rate and double-down rate over the different datasets. 
- `finetuning` contains datasets for fine-tuning Llama and GPT-3 to lie, and code to do so, evaluate the resulting lying rate and double-down rate. See `finetuning/README.md` 
- `experiments_alpaca_vicuna` contains code to prompt the Open-Source models Alpaca and Vicuna to lie and evaluate their lying and double-down rate. See `experiments_alpaca_vicuna/README.md` for more details. 

### For training and testing the lie detectors
- `lllm/questions_loader.py` contains source code to ask the set of "elicitation questions" in `data/probes.csv` after a model was suspected to have lied. This relies on the classes defined in `dialogue_classes.py`
- `lying_rate_double_down_rate_probes.ipynb` asks these elicitation questions to GPT-3.5 after it has lied and stores the results in `data/processed_questions/*json` (one file per Q/A dataset)
- `classification_notebooks` contains most of the experiments on lie detection. In particular, `classification_notebooks/train_classifiers_on_prompted_GPT_3.5.ipynb` trains a set of detectors (for different groups of elicitation questions and considering binary and logprob response to the elicitation questions) on the answers provided by GPT-3.5, which are then tested in other experiments. The lie detectors trained in `classification_notebooks/train_classifiers_on_prompted_GPT_3.5.ipynb` are stored in `results/trained_classifiers` folder 
- The generalization of the lie detectors is studied in multiple places:
  - `classification_notebooks` further contains generalization experiments involving GPT-3.5; see details in `classification_notebooks/README.md`. Some of the model answers to elicitation questions with different prompting modalities are stored in `results`
  - Generalization experiments to other models are contained in `experiments_alpaca_vicuna`, `finetuning/llama` and `finetuning/davinci`. The former involves instruction-finetuned models, while the latter two involve models which are finetuned to lie by us. See the `README.md` file in those folders for more details. Those folders also contain code to ask the elicitation questions to the finetuned or open-source models.

### Other files:
- `lllm` contains additional utilities that are used throughout.
- `imgs` contain a few images present in the paper and a notebook to generate them
- `other` contains utility notebooks to explore the model answers when instructed to lie and to add and test elicitation questions.   


## Practicalities

To use this code, create a clean `Python` environment and then run 

```pip install -r requirements.txt```

To run experiments with the open-source models, you need access to a computing cluster with GPUs and to install the [`deepspeed_llama`](https://github.com/LoryPack/deepspeed_llama) repository on that cluster. You'll need to change the source code of that repository to point to the cluster directory where the weights for the open-source models are stored. `experiments_alpaca_vicuna` and `finetuning/llama` contain a few `*.sh` example scripts for clusters using `slurm`.
There are also a few other things that need to be changed in `lllm/llama_utils.py` according to the paths of your cluster. Moreover, `finetuning/llama/llama_ft_folder.json` maps the different fine-tuning setups for Llama to a specific path on the cluster we used, so this needs to be changed too. 

Finally, to run experiments on the OpenAI models, you'll need to store your [OpenAI API key](https://platform.openai.com/account/api-keys) in a `.env` file in the root of this directory, with the format: 

```OPENAI_API_KEY=sk-<your key>```

Running experiments with the OpenAI API will incur a monetary cost. Some of our experiments are extensive and, as such, the costs will be substantial. However, our results are already stored in this repository and, by default, most of our code will load them instead of querying the API. Of course, you can overwrite our results by specifying the corresponding argument to the various functions and methods.