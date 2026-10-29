import httpx, time

def probe_openai_compat(base_url: str, model: str, timeout: float = 30.0) -> dict:
    health_ok = False; models_ok = False; chat_ok = False
    try:
        r = httpx.get(f"{base_url}/health", timeout=timeout); health_ok = r.status_code == 200
    except Exception as e: health_ok = False
    try:
        r = httpx.get(f"{base_url}/v1/models", timeout=timeout); models_ok = r.status_code == 200 and any(m["id"]==model for m in r.json().get("data",[]))
    except Exception: models_ok = False
    try:
        r = httpx.post(f"{base_url}/v1/chat/completions", json={"model": model, "messages": [{"role":"user","content":"hi"}]}, timeout=timeout); chat_ok = r.status_code==200
    except Exception: chat_ok = False
    return {"health_ok": health_ok, "models_ok": models_ok, "chat_ok": chat_ok}
