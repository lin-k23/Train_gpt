# @Time     : 2025/5/23 00:40
# @Author   : Kun Lin
# @Filename : ft_MovieDialogs.py


from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
import os
# import kagglehub # 不再需要 Kaggle Hub
import zipfile # 用于处理 zip 文件
from datasets import Dataset, DatasetDict # 用于创建 Hugging Face Dataset 对象
from transformers import TrainingArguments, Trainer
import torch # 导入 torch 以检查 GPU 可用性

# --- Configuration ---
custom_cache_dir = "../models"

# 本地数据集配置
LOCAL_ZIP_FILE_PATH = "../datasets/archive.zip" # 假设的本地 zip 文件路径
EXTRACTION_PATH = "../datasets/cornell_movie_dialogs_extracted_from_zip" # 解压目标路径
# Cornell Movie Dialogs Corpus 通常解压后会有一个子文件夹
# 常见的子文件夹名称是 "cornell movie-dialogs corpus"
# 如果你的压缩包结构不同，需要调整这个值
EXTRACTED_SUBFOLDER_NAME = "cornell movie-dialogs corpus"
MOVIE_LINES_FILENAME = "movie_lines.txt"
# movie_lines.txt 的完整路径将在解压后动态确定

output_dir = "../res"
logging_dir = '../logs'
model_name = "gpt2" # 或者 "gpt2-medium", "gpt2-large", 等.

# 创建必要的目录
if not os.path.exists(custom_cache_dir):
    os.makedirs(custom_cache_dir)
if not os.path.exists(EXTRACTION_PATH):
    os.makedirs(EXTRACTION_PATH)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)

def load_data_from_local_zip():
    """
    从本地的 archive.zip 文件解压 Cornell Movie Dialogs Corpus,
    读取 movie_lines.txt, 并返回一个 Hugging Face DatasetDict.
    """
    print(f"Attempting to process local zip file: {LOCAL_ZIP_FILE_PATH}")

    # 动态确定 movie_lines.txt 的路径
    # 假设解压后，movie_lines.txt 位于 EXTRACTION_PATH / EXTRACTED_SUBFOLDER_NAME / MOVIE_LINES_FILENAME
    prospective_movie_lines_file_path = os.path.join(EXTRACTION_PATH, EXTRACTED_SUBFOLDER_NAME, MOVIE_LINES_FILENAME)

    if not os.path.exists(LOCAL_ZIP_FILE_PATH):
        print(f"ERROR: Local zip file not found at {LOCAL_ZIP_FILE_PATH}")
        print("Please ensure 'archive.zip' is present in the '../datasets' directory.")
        exit()

    # 检查数据是否已解压
    if os.path.exists(prospective_movie_lines_file_path):
        print(f"Found existing extracted data at {prospective_movie_lines_file_path}. Skipping extraction.")
        movie_lines_file_path = prospective_movie_lines_file_path
    else:
        print(f"Extracting {LOCAL_ZIP_FILE_PATH} to {EXTRACTION_PATH}...")
        try:
            with zipfile.ZipFile(LOCAL_ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(EXTRACTION_PATH)
            print("Extraction complete.")

            # 验证 movie_lines.txt 文件在解压后是否存在
            movie_lines_file_path = prospective_movie_lines_file_path # 再次赋值，因为现在应该存在了
            if not os.path.exists(movie_lines_file_path):
                # 如果在预期的子文件夹中找不到，尝试在解压根目录查找
                fallback_movie_lines_path = os.path.join(EXTRACTION_PATH, MOVIE_LINES_FILENAME)
                if os.path.exists(fallback_movie_lines_path):
                    print(f"Note: {MOVIE_LINES_FILENAME} found directly in {EXTRACTION_PATH}, not in subfolder '{EXTRACTED_SUBFOLDER_NAME}'.")
                    movie_lines_file_path = fallback_movie_lines_path
                else:
                    print(f"ERROR: {MOVIE_LINES_FILENAME} not found at expected path: {movie_lines_file_path} after extraction.")
                    print(f"Also checked fallback path: {fallback_movie_lines_path}")
                    print(f"Contents of extraction directory ({EXTRACTION_PATH}): {os.listdir(EXTRACTION_PATH)}")
                    # 列出可能的子文件夹内容
                    if os.path.exists(os.path.join(EXTRACTION_PATH, EXTRACTED_SUBFOLDER_NAME)):
                         print(f"Contents of subfolder ({os.path.join(EXTRACTION_PATH, EXTRACTED_SUBFOLDER_NAME)}): {os.listdir(os.path.join(EXTRACTION_PATH, EXTRACTED_SUBFOLDER_NAME))}")
                    raise FileNotFoundError(f"{MOVIE_LINES_FILENAME} not found. Please check the zip file structure and extraction process.")
        except zipfile.BadZipFile:
            print(f"Error: {LOCAL_ZIP_FILE_PATH} is not a valid zip file or is corrupted.")
            exit()
        except Exception as e:
            print(f"An unexpected error occurred during extraction: {e}")
            exit()

    # 从文件加载电影对话行
    print(f"Loading movie lines from: {movie_lines_file_path}")
    lines_data = []
    # 文件通常是 ISO-8859-1 或 'latin-1' 编码
    try:
        with open(movie_lines_file_path, 'r', encoding='iso-8859-1') as f:
            for line_entry in f:
                parts = line_entry.strip().split(" +++$+++ ")
                if len(parts) == 5:
                    # 对话文本是第5部分 (索引为4)
                    lines_data.append({"text": parts[4]})
    except Exception as e:
        print(f"Error reading or parsing {movie_lines_file_path}: {e}")
        exit()

    if not lines_data:
        print(f"No data loaded from {movie_lines_file_path}. Please check the file format, content, and encoding.")
        exit()

    print(f"Successfully loaded {len(lines_data)} lines from local zip dataset.")

    hf_dataset = Dataset.from_list(lines_data)
    dataset_dict = DatasetDict({"train": hf_dataset})
    return dataset_dict

# --- 加载模型和分词器 ---
print(f"Loading model and tokenizer for {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=custom_cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir)

# 如果 GPT-2 没有 pad token，则设置它
if tokenizer.pad_token is None:
    print("Tokenizer does not have a pad token, setting it to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# --- 从本地 Zip 加载并处理数据 ---
print("Attempting to load dataset from local zip file...")
dataset = load_data_from_local_zip() # 现在返回 DatasetDict

if dataset is None or "train" not in dataset or not dataset["train"]:
    print("Failed to load or process data from local zip. Exiting.")
    exit()

# 定义最大序列长度
max_length = 512

def format_and_tokenize_function(examples):
    text_column_name = 'text'
    if text_column_name not in examples:
         raise ValueError(f"'{text_column_name}' column not found in examples. Columns available: {list(examples.keys())}")

    tokenized_output = tokenizer(
        examples[text_column_name],
        truncation=True,
        max_length=max_length,
    )
    return tokenized_output

print("Formatting and tokenizing dataset...")
train_dataset_raw = dataset["train"]

columns_to_remove = list(train_dataset_raw.column_names)

tokenized_train_dataset = train_dataset_raw.map(
    format_and_tokenize_function,
    batched=True,
    remove_columns=columns_to_remove,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)

# --- 训练设置 ---
print("Setting up training arguments...")

# 检查是否有可用的 CUDA GPU
use_fp16 = torch.cuda.is_available()
if use_fp16:
    print("CUDA is available. Enabling fp16 (mixed precision) training for PyTorch acceleration.")
else:
    print("CUDA is not available. fp16 training will be disabled. Training will run on CPU and might be slow.")

# **重要提示：关于多 GPU 训练**
# 如果你有多张 GPU 并希望使用它们进行分布式训练 (推荐使用 DistributedDataParallel - DDP):
# 1. 确保你的 PyTorch 和 CUDA 环境已正确配置以识别所有 GPU.
# 2. 使用 `torchrun` (PyTorch 1.9+) 或 `python -m torch.distributed.launch` 来启动你的训练脚本.
#    例如，如果你有 8 张 GPU，使用命令:
#    `torchrun --nproc_per_node=8 your_script_name.py`
#    (将 `your_script_name.py` 替换为你的 Python 文件名)
# 3. `per_device_train_batch_size` 指的是每张卡的批量大小。
#    总的有效批量大小将是 `per_device_train_batch_size * 你的GPU数量 * gradient_accumulation_steps`.
#
# **关于 CUDA Out of Memory (OOM) 错误**:
# OOM 错误可能是由于以下原因造成的:
#   - `per_device_train_batch_size` 对于单张 GPU 的可用显存来说过大 (特别是如果其他进程也在使用该 GPU).
#   - 模型本身较大，或者序列长度 (`max_length`) 较长.
# 缓解 OOM 的策略:
#   - 减小 `per_device_train_batch_size`.
#   - 增大 `gradient_accumulation_steps` 以保持较大的有效批量大小，同时减少单步的显存占用.
#   - 确保没有其他不必要的进程占用大量 GPU 显存.
#   - 尝试使用 `fp16=True` (混合精度训练)，如果 GPU 支持.

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # 减小每个设备的批量大小以降低显存占用
    gradient_accumulation_steps=4,  # 增大梯度累积步数，有效批量大小 = 1 * num_gpus * 4
                                    # 如果有 8 张卡，则有效批量大小为 1 * 8 * 4 = 32
    save_strategy="steps",
    save_steps=5000,
    save_total_limit=2,
    logging_dir=logging_dir,
    logging_steps=500,
    eval_strategy="no",
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    fp16=use_fp16, # 启用混合精度训练 (如果 CUDA 可用)
                  # 这利用 PyTorch 的 AMP (Automatic Mixed Precision) 来加速
    report_to="none", # 禁用 wandb 或其他报告后端，避免不必要的提示
    # 如果你使用的是 PyTorch 2.0 或更高版本，并且希望进一步加速，可以考虑以下选项：
    # torch_compile=True, # 尝试使用 torch.compile 进行模型编译优化 (实验性)
    # torch_compile_backend="inductor", # 可以指定编译后端
    # torch_compile_mode="reduce-overhead", # 或 "max-autotune"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

# --- 训练并保存 ---
print("Starting training...")
try:
    trainer.train()
except Exception as e:
    print(f"An error occurred during training: {e}")
    raise

final_save_path = os.path.join(custom_cache_dir, "cornell_movie_dialog_gpt2_finetuned_local_zip")
print(f"Saving fine-tuned model to {final_save_path}...")
trainer.save_model(final_save_path)
tokenizer.save_pretrained(final_save_path)

print("Training complete and model saved.")
