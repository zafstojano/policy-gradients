import argparse
import json

from torch.utils.data import DataLoader


def get_dataloader(
    dataset_path: str,
    prompts_per_step: int,
    max_dataset_size: int | None = None,
    max_num_terms: int = 3,
    max_num_digits: int = 3,
    max_question_len: int = 128,
) -> DataLoader:
    predicate = (
        lambda x: x["num_terms"] <= max_num_terms
        and x["num_digits"] <= max_num_digits
        and len(x["question"]) < max_question_len
    )
    with open(dataset_path) as f:
        dataset = [json.loads(line) for line in f]
    dataset = [x for x in dataset if predicate(x)]
    if max_dataset_size is not None:
        dataset = dataset[:max_dataset_size]

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=prompts_per_step,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    return dataloader


def main(args):
    dataloader = get_dataloader(
        dataset_path=args.dataset_path, prompts_per_step=args.prompts_per_step, max_dataset_size=16
    )

    for step, batch in enumerate(dataloader):
        print(f"Step {step}: {batch}")
        print("-"*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/math_tasks.jsonl")
    parser.add_argument("--prompts_per_step", type=int, default=8)
    args = parser.parse_args()

    main(args)
