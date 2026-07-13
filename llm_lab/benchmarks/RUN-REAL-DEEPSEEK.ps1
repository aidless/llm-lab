$env:OPENAI_API_KEY = "sk-your-deepseek-key"
$env:LLM_BASE_URL = "https://api.deepseek.com/v1"
$env:LLM_MODEL = "deepseek-chat"

cd F:\TMLR\llm_lab
python benchmarks\self_bench.py --mode all --steps 50 --real --output benchmarks\v2-results.json
