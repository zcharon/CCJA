import numpy as np
import argparse
import random
import sys
sys.path.insert(0, './GPT2ForwardBackward')
from nltk.corpus import stopwords
from util import *
import torch
stop_words = set(stopwords.words('english'))


def get_args():
    parser = argparse.ArgumentParser()
    ## setting
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--pretrained_model", type=str, default='bert')
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--mlm_model", default='bert')
    parser.add_argument("--perturb_layer_index", default=7)
    parser.add_argument("--behavior_nums", type=int, default=10)
    parser.add_argument("--devices", type=str, default='cuda:0')
    parser.add_argument("--mlm_device", type=str, default='cuda:0')
    parser.add_argument("--lr", default=0.1)

    ## experiment
    parser.add_argument("--save_file_path", type=str, default='file.csv')
    parser.add_argument("--save_file_name", type=str, default='CCJA-Llama2')
    parser.add_argument("--start", type=int, default=0, help="loading data from ith examples.")
    parser.add_argument("--end", type=int, default=10, help="loading data util ith examples.")
    parser.add_argument("--mode", type=str, default='prefix', 
                        choices=['prefix', 'multi_behavior', 'multi_agent'])
    
    # iterations
    parser.add_argument("--repeat_nums", type=str, default=3)
    parser.add_argument("--iter_nums", type=str, default=1200)

    # gaussian noise
    parser.add_argument("--init_mag", type=float, default=3)

    args = parser.parse_args()
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    args = get_args()
    
    model_path_dicts = {"llama-2-7b-chat-hf": "../huggingface_cache/llama/Llama-2-7b-chat-hf/",
                    "llama-3-8B-Instruct": "../huggingface_cache/llama/llama-3-8B-Instruct/",
                    "vicuna-7b-v1.5": "../huggingface_cache/vicuna/vicuna-7b-v1.5/",
                    "vicuna-13b-v1.5": "../huggingface_cache/vicuna/vicuna-13b-v1.5/",
                    "guanaco-13B-HF": "../huggingface_cache/guanaco/guanaco-13B-HF/",
                    "mistral-7B-Instruct-v0.2": "../huggingface_cache/mistralai/Mistral-7B-Instruct-v0.2/",
                    "mistral-7B-Instruct-v0.3": "../huggingface_cache/mistralai/Mistral-7B-Instruct-v0.3/"}
    if 'bert' in args.mlm_model:
        mlm_path = "../huggingface_cache/bert-base-uncased/"

    if args.seed != -1: 
        seed_everything(args.seed)

    if "multi_agent" in args.mode:
        model_names = args.pretrained_model.split('&')
        model_paths = [model_path_dicts[mn] for mn in model_names]
        devices = args.devices.split('&')
        mlm_model, mlm_tokenizer = load_mlm_model_and_tokenizer(mlm_path, device=devices[0])
        mlm_model.eval()
        models, tokenizers, mapping_tensors = [], [], []
        for model_path, dev in zip(model_paths, devices):
            model, tokenizer = load_model_and_tokenizer(model_path, low_cpu_mem_usage=True, use_cache=False, device=dev)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            models.append(model)
            tokenizers.append(tokenizer)
            mapping_tensors.append(get_mapping_tensor(mlm_tokenizer, tokenizer).to(dev))
        chatformats = [ChatFormat(mn) for mn in model_names]
        from attack_multi_agent import attack_generation
        attack_generation(models, mlm_model, tokenizers, mlm_tokenizer, devices, args, mapping_tensors=mapping_tensors, 
                          chatformats=chatformats, model_names=model_names)
    else:
        model_path = model_path_dicts[args.pretrained_model]
        mlm_model, mlm_tokenizer = load_mlm_model_and_tokenizer(mlm_path, device=args.mlm_device) 
        mlm_model.eval()
        model, tokenizer = load_model_and_tokenizer(model_path, low_cpu_mem_usage=True, use_cache=False, device=args.devices)
        mapping_tensor = get_mapping_tensor(mlm_tokenizer, tokenizer).to(args.devices)
        chatformat = ChatFormat(args.pretrained_model)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        if "prefix" in args.mode:
            from attack_prefix import attack_generation
            attack_generation(model, mlm_model, tokenizer, mlm_tokenizer, args, mapping_tensor=mapping_tensor, 
                            chatformat=chatformat)
        elif "multi_behavior" in args.mode:
            from attack_multi_behavior import attack_generation
            attack_generation(model, mlm_model, tokenizer, mlm_tokenizer, args.devices, args, mapping_tensor=mapping_tensor, 
                            chatformat=chatformat)

if __name__ == "__main__":
    main()