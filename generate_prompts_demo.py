from lllm.questions_loaders import Questions1000
from utils.modelUtils import *
from utils.utils import *

dataset = Questions1000()
# dataset.truth_instructions
# dataset.lie_instructions
model_name = "/common2/public/LLAMA2-HF/Llama-2-7b-chat-hf"
lie_instruction_num = "random"

questions = dataset["question"]
answers = dataset["answer"]
if_can_answers = dataset[model_name + "_can_answer"]

all_layerAIE_orig = []
all_layerAIE_lie = []
all_layerAIE_lie_attempt = []
all_kurt_orig = []
all_kurt_lie = []
all_kurt_lie_attempt = []

success_lie_instruct = 0
attempt_num = 0
for k, question in enumerate(tqdm(questions)):  # for question in tqdm(questions):
    expected_answer = answers[k]
    if_can_answer = if_can_answers[k]
    if bool(if_can_answer) == False:
        print("-->model can't answer such question")
        continue
    attempt_num += 1

    prompt_orig = prepare_prompt(question, dataset.truth_instructions[0])
    ########## todo ##########
    answer = Generate_model_answer(prompt_orig)  # example of generating model answer function
    # get AIE for prompt_orig
    # if you try other model, and there is no dataset[model_name + "_can_answer"] in the original dataset, you can do:
    # can_answer_orig = dataset._is_answer_correct(
    #     answer, expected_answer, question
    # )
    if can_answer_orig == True:
        # model can answer
        # ...
    else:
        continue

    lie_instructions = dataset.lie_instructions
    if isinstance(lie_instruction_num, int):
        num = lie_instruction_num
    elif lie_instruction_num == "random":
        num = random.randint(0, len(lie_instructions) - 1)
    if "*out loud*" in lie_instructions[num]:
        new_prompt = prepare_prompt(question, lie_instructions[num])
        prompt_lie = new_prompt
    else:
        prompt_lie = generate_input(conv_template, lie_instructions[num] + question)

    ########## todo ##########
    answer = Generate_model_answer(prompt_lie) # example of generating model answer function
    # get AIE for prompt_orig
    # ...

    if "*out loud*" in answer:
        answer = answer.split("*out loud*")[1].strip()
        if "Question:" in answer:
            answer = answer.split("Question:")[0].strip()
    # check if the generated answer cover expected_answer
    can_answer_lie = dataset._is_answer_correct(
        answer, expected_answer, question
    )
    print("-->expected_answer", expected_answer)
    print("-->can_answer after lie", can_answer_lie)
    # if bool(can_answer_lie) == True:
    # print("-->can_answer after lie", can_answer_lie)
    # print("-->expected_answer", expected_answer)
    # print("-->answer", answer)

    if bool(can_answer_lie) == False:
        # we only consider those sample pairs. 1. prompt_orig which the model can answer correctly; 2. prompt_lie which make the model lie successfully

print("-->Success lie instruct rate: {} ({}/{})".format(...)

########## todo ##########
# lie detector training
