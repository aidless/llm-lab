$env:OPENAI_API_KEY = "sk-your-deepseek-key"
$env:LLM_BASE_URL = "https://api.deepseek.com/v1"
# deepseek-chat and deepseek-reasoner are being deprecated 2026-07-24 23:59
# Beijing time. Use deepseek-v4-flash (non-thinking mode) or
# deepseek-v4-flash with thinking_mode enabled.
$env:LLM_MODEL = "deepseek-v4-flash"
# DeepSeek v4 defaults to internal reasoning pass. That inflates
# both latency and token count. Set this to disable it (the
# worker forwards an extra_body={thinking: {type: disabled}}
# to the chat.completions endpoint).
$env:DISABLE_LLM_THINKING = "1"

cd F:\TMLR\llm_lab
python benchmarks\self_bench.py --mode all --steps 50 --real --output benchmarks\v2-results.json
