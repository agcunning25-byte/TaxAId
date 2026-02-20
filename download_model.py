from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
    filename="Phi-3-mini-4k-instruct-q4.gguf",
    local_dir="./models",
    local_dir_use_symlinks=False
)

print("Model downloaded to:", model_path)