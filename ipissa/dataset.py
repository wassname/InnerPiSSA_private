import dataclasses
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase



@dataclasses.dataclass
class DatasetEntry:
    positive: str
    negative: str


def make_dataset(
    template: str,
    positive_personas: list[str],
    negative_personas: list[str],
    suffix_list: list[str],
    tokenizer: PreTrainedTokenizerBase,
    verbose: bool= False,
) -> list[DatasetEntry]:
    dataset = []
    for suffix in suffix_list:
        for positive_persona, negative_persona in zip(
            positive_personas, negative_personas
        ):

            positive_prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': template.format(persona=positive_persona)},
                    {'role': 'assistant', 'content': suffix}],
                tokenize=False,
                continue_final_message=True
            )
            negative_prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': template.format(persona=negative_persona)},
                    {'role': 'assistant', 'content': suffix}],
                tokenize=False,
                continue_final_message=True,

            )
            dataset.append(
                DatasetEntry(
                    positive=positive_prompt,
                    negative=negative_prompt,
                )
            )
    if verbose:
        for i in range(3):
            print(f"Example {i+1}:")
            print(f"Positive: {dataset[i].positive}")
            print(f"Negative: {dataset[i].negative}")
    return dataset
