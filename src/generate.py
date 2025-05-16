# adapted: https://medium.com/@geronimo7/llms-multi-gpu-inference-with-accelerate-5a8333e4c5db

from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import argparse
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
from peft import LoraConfig, PeftModel

import warnings
warnings.filterwarnings("ignore")

kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
accelerator = Accelerator(kwargs_handlers=[kwargs])

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--tokenizer', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='generated/iter0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--input_dir', type=str, default='./formatted')
    parser.add_argument('--split', type=str, default='train')
    # Arguments for LoraModel
    # base_model -- model for adapter insertion
    # lora_r, lora_alpha -- parameters in LoraConfig
    # if you want to specify LoraConfig in detail, just change LoraConfig in main
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--lora_r', type=int, default=0)
    parser.add_argument('--lora_alpha', type=int, default=0)
    return parser.parse_args()

def prepare_prompts(prompts, tokenizer, batch_size=4):
    """Prepare prompts for tokenization."""
    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    batches_tok=[]
    tokenizer.padding_side="left"
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding='longest',
                truncation=False,
                pad_to_multiple_of=8,
                add_special_tokens=False).to("cuda")
            )
    tokenizer.padding_side="right"
    return batches_tok

def main():
    args = parse_arguments()
    model_path = args.model
    tokenizer_path = args.tokenizer
    data_frac = args.data_frac
    batch_size = args.batch_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load a base model and tokenizer
    if args.lora_r == 0:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": accelerator.process_index},
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # Specify lora adapters
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(args.base_model),
            model_path,
            config=lora_config
        ).base_model.merge_and_unload().to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # load data
    data = load_dataset(args.input_dir, split=args.split)
    data = data.shuffle(seed=42)
    if args.frac_len > 0:
        sub_len = args.frac_len
        if sub_len*(data_frac+1) > len(data):
            data = data[sub_len*data_frac:]['real']
        else:
            data = data[sub_len*data_frac:sub_len*(data_frac+1)]['real']
    else:
        data = data[:]['real']

    prompts_all = ["### Instruction: " + data[idx][0]['content'] + "\n\n### Response: " for idx in range(len(data))]
    prompts_old = [data[idx][0]['content'] for idx in range(len(data))]
    corrects_all = [data[idx][1]['content'] for idx in range(len(data))]

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()
    start=time.time()

    # divide the prompt list onto the available GPUs
    with accelerator.split_between_processes(prompts_all) as prompts:
        results = []
        prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=args.batch_size)

        for prompts_tokenized in tqdm(prompt_batches):
            # set max_new_tokens smaller for faster inference
            outputs_tokenized=model.generate(**prompts_tokenized, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)

            # remove prompt from gen. tokens
            outputs_tokenized=[ tok_out[len(tok_in):]
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ]
            # decode gen. tokens
            outputs=tokenizer.batch_decode(outputs_tokenized)
            results.extend(outputs)

    # collect results from all the GPUs and remove paddings
    results_gathered=gather_object(results)
    results = [r.replace("</s>","").lstrip() for r in results_gathered]

    if accelerator.is_local_main_process:
        timediff=time.time()-start
        print(f"time elapsed: {timediff}")

        # collecting data
        for idx in range(len(corrects_all)):
            d = {"real": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": corrects_all[idx]}], "generated": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": results[idx]}]}
            if args.split == 'test':
                filename = f"{args.output_dir}/loser_{data_frac}_test.jsonl"
            else:
                filename = f"{args.output_dir}/loser_{data_frac}.jsonl"
            with open(filename, 'a') as f:
                json.dump(d, f)
                f.write('\n')


if __name__ == "__main__":
    main()