$env:OPENAI_API_KEY = "sk-your-deepseek-key"
$env:LLM_BASE_URL = "https://api.deepseek.com/v1"
$env:LLM_MODEL = "deepseek-v4-flash"
# Disable thinking mode for cleaner benchmark numbers
$env:DISABLE_LLM_THINKING = "1"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
python benchmarks\self_bench.py --mode all --steps 50 --real --output benchmarks\v2-results.json
