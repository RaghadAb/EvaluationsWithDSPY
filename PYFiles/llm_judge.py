# refactored the code for Judge metric
from pydantic import BaseModel, Field
from openai import OpenAI
import dspy
# pyright: reportUnusedImport=false
# pyright: reportMissingImports=false


class Evaluation(BaseModel):
    explanation: str = Field(
        ..., description="A detailed text evaluation of the answer."
    )
    accuracy: int = Field(..., description="Score for accuracy, either 0 or 1.")
    clarity: int = Field(..., description="Score for clarity, either 0 or 1.")
    completeness: int = Field(..., description="Score for completeness, either 0 or 1.")


class EvalOutput(BaseModel):
    evaluations: Evaluation = Field(
        ...,
        description="An evaluation containing explanation and various criteria scores.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "evaluation": {
                    "explanation": "The answer provided is very satisfactory, meeting some of the criteria.",
                    "accuracy": 1,
                    "clarity": 0,
                    "completeness": 1,
                }
            }
        }


def gpt4_judge_metric(pred: dspy.Prediction, example: dspy.Example, trace=None):
    judge_prompt = """You are an expert evaluator responsible for assessing whether a given answer meets the specified evaluation criteria.

    You will receive the following inputs:
    
    A question.
    The evaluation criteria.
    The model's answer to evaluate.
    
    Instructions:
    Evaluation: Criterias has been provided below, evaluate the given answer based on the list.
    '
    CRITERIA
    
    accuracy: Are the facts correct and aligned with email security standards (e.g., DMARC, SPF, DKIM)?
    clarity: Is the response concise, well-structured, and easy to understand?
    completeness: Does it fully address the question and cover all relevant aspects?
    
    Response Format:
    A dictionarie with the following structure:
    explanation: Provide a detailed text evaluation of how the answer meets or fails to meet each criterion.
    accuracy: Provide only the number (either 1 or 0). Do not include any Explanation.
    clarity: Provide only the number (either 1 or 0). Do not include any Explanation.
    completeness: Provide only the number (either 1 or 0). Do not include any Explanation.
    
    Question: {question} \n\n
    Answer: {answer} \n\n
    """
    judge_input = judge_prompt.format(question=example.answer, answer=pred.answer)

    # Use OpenAI API for GPT-4
    client = OpenAI()  # OPENAI_API_KEY
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": judge_input,
            }
        ],
        temperature=0,
    )

    # Parse the response content
    judge_output = response.choices[0].message.content
    # print(judge_output)
    # Validate the response with Evaluation model
    output_obj = Evaluation.model_validate_json(judge_output)

    # Extract and print evaluations
    # print(output_obj)

    # Return the Accuracy score, or extend to return Clarity and Completeness
    return output_obj.accuracy
