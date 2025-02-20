# - Start python server,
# - Pull model
# pyright reportUnusedImport=false
# ruff noqa: F401
# pyright: reportMissingImports=false

import dspy
from dspy.evaluate import Evaluate
import pandas as pd
from time import time
import click  # New import for click


def llm_config(lm: dspy.LM):
    data = lm.kwargs
    data["model"] = lm.model
    data["model_type"] = lm.model_type
    return data


print("Script will do the following: ")
print("Ensure model that you want to test loaded in Ollama")
print("""
- Setup up DSPy inference for the model based on 
      temperature & tokens 
- Send in Data
- Setup the Evaluator 
- Check Evaluator for 5 data points
      """)


@click.command()
@click.option('--model_name', default="ollama_chat/llama3.2:1b", help='Provide model name like ollama_chat/llama3.2:1b')  # Default value
@click.option('--token_count', default=150, type=int, help='Provide token count as integer')  # Default value
@click.option('--temperature', default=0.2, type=float, help='Provide temperature as float (0.0 to 1.0)')  # Default value
@click.option('--module_name', default="predict", help='Provide the module to use in program (predict or cot)')  # Default value
def main(model_name, token_count, temperature, module_name):
    # get model name:
    model_name = model_name.strip().replace(" ", "")
    # model_name = "ollama_chat/llama3.2:1b"
    # token_count = 50
    # temperature = 0.2

    ollama_server = "http://127.0.0.1:11434"

    lm = dspy.LM(
        model=model_name,
        api_base=ollama_server,
        api_key="",
        temperature=temperature,
        max_tokens=token_count,
    )

    dspy.configure(lm=lm)


    class Recommender(dspy.Signature):
        """Provide Recommendation for the given question from the domain of Email, Network
        Monitoring and Management."""

        question: str = dspy.InputField(
            desc="Questions or situation on Email, Network or Domain related Configuration"
        )
        answer: str = dspy.OutputField(desc="Recommendation for the question or situation")


    def recommend_answer_metric(
        example: dspy.Example, prediction: dspy.Prediction, trace=None
    ) -> bool:
        """
        Evaluates correctness of recommendation answers predicted.

        Args:
            example (Example): The dataset example containing expected people entities.
            prediction (Prediction): The prediction from the DSPy people extraction program.
            trace: Optional trace object for debugging.

        Returns:
            bool: True if predictions match expectations, False otherwise.
        """
        return prediction.answer == example.answer


    class RecommenderCOT(dspy.Signature):
        """Provide Recommendation for the given question from the domain of Email, Network
        Monitoring and Management."""

        question: str = dspy.InputField(
            desc="Questions or situation on Email, Network or Domain related Configuration"
        )
        answer: str = dspy.OutputField(desc="Recommendation for the question or situation")

    def recommend_answer_metric_cot(
        example: dspy.Example, prediction: dspy.Prediction, trace=None
    ) -> bool:
        """
        Evaluates correctness of recommendation answers predicted.

        Args:
            example (Example): The dataset example containing expected people entities.
            prediction (Prediction): The prediction from the DSPy people extraction program.
            trace: Optional trace object for debugging.

        Returns:
            bool: True if predictions match expectations, False otherwise.
        """
        return prediction.answer == example.answer


    # Loading the data from "cleaned_concatenated.csv"
    data = pd.read_csv("cleaned_concatenated.csv")
    print("** Data Loaded Successfully ** \n")
    # print("Data head: ")
    # print(data.head(2))

    # preparing the example

    qa_dataset = []
    for dp in data.to_dict(orient="records"):
        qa_dataset.append(
            dspy.Example(question=dp["Question"], answer=dp["Recommendation"]).with_inputs(
                "question"
            )
        )
    # print(type(qa_dataset[0]))
    # print(qa_dataset[4])
    print("** Data Prepared Successfully **\n")
    print("** Data Size: ", len(qa_dataset), "**\n")

    # Set up the evaluator, which can be re-used in your code.
    evaluator = Evaluate(
        devset=qa_dataset[:10],
        display_progress=False,
        display_table=5,
        provide_trace=True,
    )
    if module_name == "predict":
        print("Using Predict Module")
        program = dspy.Predict(Recommender)
        print("** Evaluator Setup Successfully **\n")
        st = time()
        evaluator(program=program, metric=recommend_answer_metric)
        time_taken = round(time() - st, 1)

        print(f"Time taken for predict evaluator to run: {time_taken}s")
    elif module_name == "cot":
        print("Using Chain of Thought Module")
        program_cot = dspy.ChainOfThought(RecommenderCOT)
        print("** Evaluator Setup Successfully **\n")
        st = time()
        evaluator(program=program_cot, metric=recommend_answer_metric_cot)
        time_taken = round(time() - st, 1)
        print(f"Time taken for cot evaluator to run: {time_taken}s")
    else:
        print("Invalid module name")
        return

    # print("program: ", program)
    # print(qa_dataset[0].inputs())
    # output = program(**qa_dataset[0].inputs())
    # print(f"Output: {output}")

    

    # Show the configuration details
    print("*** LLM Configuration is ***\n")
    lm_config = llm_config(lm)
    print(lm_config, end="\n\n")


if __name__ == '__main__':
    main()  # Call the main function
