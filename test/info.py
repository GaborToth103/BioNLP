from huggingface_hub import hf_hub_download, snapshot_download, scan_cache_dir

# Scan the local cache for all files
cache_info = scan_cache_dir()

# Print useful info
for repo in cache_info.repos:
    print(f"Repository: {repo.repo_id}")
    print(f"Size: {repo.size_on_disk / 1e6:.2f} MB")
    print("------------")