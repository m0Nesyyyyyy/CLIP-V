from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2-VL-2B-Instruct', cache_dir='/home/liweiyan/workspace/Tip-Adapter/qwen2-vl/', revision='master')
