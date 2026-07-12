import os

os.environ["LLM_PROVIDER"] = "ollama"
os.environ["LLM_BASE_URL"] = "http://localhost:11434/v1"
os.environ["LLM_MODEL"] = "deepseek-r1:7b"
os.environ["LLM_API_KEY"] = "ollama"

from llm_lab import worker as wrk

r = wrk.call_llm("Say hello in one word")
print("finish:", r["finish_reason"])
print("model:", r["model"])
print("cost:", r["cost_usd"])
print("tokens:", r["token_usage"])
out = r["output"]
print("output length:", len(out))
print("output preview:", out[:150].encode("ascii", errors="replace").decode())
