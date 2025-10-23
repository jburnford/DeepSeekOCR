# Optional: Hugging Face Setup

## TL;DR

**DeepSeek-OCR does NOT require a Hugging Face API key.** The model is publicly available under MIT license and can be downloaded without authentication.

However, using a Hugging Face token can provide benefits for other tasks and is considered a best practice.

## Why Use a Hugging Face Token (Optional)

Even though not required for DeepSeek-OCR, a token provides:

1. **Higher Rate Limits** - Avoid throttling on public endpoints
2. **Access to Gated Models** - If you want to test other OCR models
3. **Model Upload/Push** - If you want to share fine-tuned models
4. **Better Error Messages** - More detailed download feedback
5. **Private Model Access** - If working with restricted datasets

## Setup (Optional)

### 1. Create Hugging Face Account

Visit https://huggingface.co/join

### 2. Generate Access Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it (e.g., "nibi-cluster-ocr")
4. Select "read" permission (sufficient for model downloads)
5. Copy the token (starts with `hf_...`)

### 3. Configure on Nibi Cluster

**Option A: Environment Variable (Recommended for Cluster)**

Add to your SLURM script:

```bash
# In scripts/install_deepseek.sh or scripts/nibi_deepseek_test.sh
export HF_TOKEN="hf_your_token_here"
```

**Option B: Login via CLI**

```bash
# Interactive session
source ~/projects/def-jic823/deepseek_venv/bin/activate
huggingface-cli login
# Paste your token when prompted
```

**Option C: Configuration File**

```bash
# Create config file
mkdir -p ~/.huggingface
cat > ~/.huggingface/token << EOF
hf_your_token_here
EOF
chmod 600 ~/.huggingface/token
```

### 4. Test Authentication (Optional)

```python
from huggingface_hub import whoami

try:
    info = whoami()
    print(f"Logged in as: {info['name']}")
except Exception as e:
    print("Not logged in (but that's OK for DeepSeek-OCR!)")
```

## Security Best Practices

### On Shared Clusters

**DON'T:**
- ❌ Commit tokens to Git
- ❌ Put tokens in world-readable files
- ❌ Share tokens in Slack/email
- ❌ Use write tokens unless necessary

**DO:**
- ✅ Use read-only tokens
- ✅ Set file permissions: `chmod 600 ~/.huggingface/token`
- ✅ Use environment variables in SLURM scripts
- ✅ Regenerate tokens if compromised

### Environment Variable Method

Update `scripts/install_deepseek.sh` and `scripts/nibi_deepseek_test.sh`:

```bash
# Add after module loading
if [ -n "$HF_TOKEN" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "Using Hugging Face authentication"
else
    echo "No HF token (OK for public models like DeepSeek-OCR)"
fi
```

Then submit with:
```bash
export HF_TOKEN="hf_your_token_here"
sbatch scripts/nibi_deepseek_test.sh
```

## Troubleshooting

### Downloads Work Without Token

**This is normal!** DeepSeek-OCR is publicly available.

### Rate Limit Errors

If you see errors like:
```
HTTPError: 429 Client Error: Too Many Requests
```

**Solution:** Add a Hugging Face token (see above)

### Slow Downloads

**Not authentication related.** Possible causes:
- Network congestion
- Large model size (3B parameters = several GB)
- Cluster internet bandwidth limits

**Solution:** Downloads happen once during first model load, then cached locally.

## Model Caching

### Default Cache Location

```bash
~/.cache/huggingface/hub/
```

### Check Cached Models

```bash
du -sh ~/.cache/huggingface/hub/
ls -lh ~/.cache/huggingface/hub/
```

### Clear Cache (if needed)

```bash
# Remove all cached models
rm -rf ~/.cache/huggingface/hub/

# Remove specific model
rm -rf ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR
```

### Change Cache Location

If your home directory has limited space:

```bash
# In SLURM scripts
export HF_HOME="/scratch/$USER/huggingface_cache"
export TRANSFORMERS_CACHE="/scratch/$USER/huggingface_cache"
```

## Testing Other OCR Models

If you want to compare with other Hugging Face models:

### GOT-OCR2.0 (May Require Token)

```python
from transformers import AutoModel

# Some models may be gated
model = AutoModel.from_pretrained(
    "stepfun-ai/GOT-OCR2_0",
    trust_remote_code=True,
    use_auth_token=True  # Uses your configured token
)
```

### Check Model Access Requirements

Before downloading:

1. Visit model page on Hugging Face
2. Look for "This is a gated model" banner
3. If gated, click "Agree and access"
4. Then token will work for that model

## Summary

For **DeepSeek-OCR specifically**:
- ✅ No token required
- ✅ Works immediately
- ✅ MIT licensed
- ✅ Public downloads

**When to add token:**
- Testing multiple models
- Hitting rate limits
- Want better error messages
- Working with gated models

## References

- [Hugging Face Tokens Docs](https://huggingface.co/docs/hub/security-tokens)
- [Model Cards](https://huggingface.co/docs/hub/model-cards)
- [DeepSeek-OCR Model Page](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
