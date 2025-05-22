# @Time     : 2025/5/22 01:37
# @Author   : Kun Lin
# @Filename : download_tool.py


from datasets import load_dataset
import os


# --- Load Dataset: trl-lib/ultrafeedback_binarized ---
# dataset_name = "trl-lib/ultrafeedback_binarized"
# split_name = "train"
# dataset_save_path = "../datasets/ultrafeedback_binarized_1"  # 定义本地保存路径
#
# # 加载指定的数据集分割
# dataset_split = load_dataset(dataset_name, split=split_name)
#
# # 保存到磁盘
# dataset_split.save_to_disk(dataset_save_path)
#
# print(f"数据集 '{dataset_name}' 的 '{split_name}' 部分已保存到: {dataset_save_path}")

# from datasets import load_dataset
# import os
# cache_dir="../datasets/lmsys_chat"
# os.makedirs(cache_dir,exist_ok=True)
# print("downloading dataset")
# # Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("lmsys/lmsys-chat-1m",cache_dir="../datasets/lmsys_chat")
# print("Done!")
# -------------------------------------------------------


# --- Load Model and Tokenizer ---
# from transformers import AutoModelForCausalLM, AutoTokenizer
# custom_cache_dir="../models"
# model_name = "gpt2-medium" # Or "gpt2-medium", "gpt2-large", etc.

# print(f"Loading model and tokenizer for {model_name}...")
# model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir)
# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir)
#
# print("download successfully!")

# from datasets import load_dataset
# import os
# cache_dir="../datasets/wild_chat"
# os.makedirs(cache_dir,exist_ok=True)
# print("downloading dataset")
# ds = load_dataset("allenai/WildChat-1M",cache_dir=cache_dir)
# print("Done")
# ---------------------------------

# --- Load Dataset: m-a-p/CodeFeedback-Filtered-Instruction ---
cache_dir="../datasets/code_feedback"
os.makedirs(cache_dir,exist_ok=True)
print("downloading dataset")
ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction",cache_dir=cache_dir)
print("Done")
# -------------------------------------------------------------