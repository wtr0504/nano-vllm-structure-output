from enum import Enum
import os

from pydantic import BaseModel
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

from nanovllm.sampling_params import StructuredOutputsParams
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Structured outputs by JSON using Pydantic schema
class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType

json_schema = CarDescription.model_json_schema()
structured_outputs_params_json = StructuredOutputsParams(json=json_schema)
sampling_params_json = SamplingParams(
    temperature=0.1,
    max_tokens=64,
    structured_outputs=structured_outputs_params_json, 
)
prompt_json = [
    "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
    # "please Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
]


def main():
    path = os.path.expanduser("/data/taoran/models/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # sampling_params = SamplingParams(temperature=0.6, max_tokens=1024 * 2)
    # prompts = [
    #     "请你用100字简单讲述一下三顾茅庐的故事",
    #     "请你用100字简单讲述一下空城计的故事",
    # ]
    
    # prompts = [
    #     tokenizer.apply_chat_template(
    #         [{"role": "user", "content": prompt}],
    #         tokenize=False,
    #         add_generation_prompt=True,
    #     )
    #     for prompt in prompt_json
    # ]
    outputs = llm.generate(prompt_json, [sampling_params_json] * len(prompt_json))

    for prompt, output in zip(prompt_json, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
