"""Tests for settings.py — Settings via pydantic-settings."""

import pytest
from llm_lab.settings import Settings, get_settings

# pydantic-settings 2.14.2 gives env vars priority over constructor args when
# aliases exist, AND reads from .env file (which has user values).  Reset
# every known env var to its Field default so tests control Settings explicitly.
_ENV_DEFAULTS = {
    "LLM_PROVIDER": "openai",
    "LLM_MODEL": "gpt-4o",
    "LLM_API_KEY": "",
    "LLM_BASE_URL": "",
    "LLM_API_KEY_2": "",
    "LLM_BASE_URL_2": "",
}


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    for var, val in _ENV_DEFAULTS.items():
        monkeypatch.setenv(var, val)


class TestSettingsProperties:
    def test_is_local_for_local_providers(self, monkeypatch):
        for provider in ("ollama", "llamacpp", "vllm", "localai", "tgi", "promptfoo"):
            monkeypatch.setenv("LLM_PROVIDER", provider)
            s = Settings()
            assert s.is_local is True, f"{provider} should be local"

    def test_is_local_for_remote_providers(self, monkeypatch):
        for provider in ("openai", "anthropic", "claude", "gemini", "google"):
            monkeypatch.setenv("LLM_PROVIDER", provider)
            s = Settings()
            assert s.is_local is False, f"{provider} should not be local"

    def test_is_local_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "OLLAMA")
        s = Settings()
        assert s.is_local is True

    def test_is_local_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "  ollama  ")
        s = Settings()
        assert s.is_local is True

    def test_is_anthropic(self, monkeypatch):
        for provider in ("anthropic", "claude"):
            monkeypatch.setenv("LLM_PROVIDER", provider)
            s = Settings()
            assert s.is_anthropic is True

    def test_is_anthropic_false(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        s = Settings()
        assert s.is_anthropic is False

    def test_is_gemini(self, monkeypatch):
        for provider in ("gemini", "google"):
            monkeypatch.setenv("LLM_PROVIDER", provider)
            s = Settings()
            assert s.is_gemini is True

    def test_is_gemini_false(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        s = Settings()
        assert s.is_gemini is False


class TestSettingsDefaults:
    # _clear_env already cleared LLM_PROVIDER / LLM_MODEL so defaults apply.
    def test_default_provider_is_openai(self):
        s = Settings()
        assert s.llm_provider == "openai"

    def test_default_model_is_gpt4o(self):
        s = Settings()
        assert s.llm_model == "gpt-4o"

    def test_deepeval_disabled_by_default(self):
        s = Settings()
        assert s.deepeval_enabled is False

    def test_default_verifier_is_deepeval(self):
        s = Settings()
        assert s.default_verifier == "deepeval"


class TestSettingsApiKeyFor:
    def test_api_key_for_no_suffix_returns_primary(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "key-primary")
        s = Settings()
        assert s.api_key_for() == "key-primary"

    def test_api_key_for_with_suffix_returns_secondary(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY_2", "key-secondary")
        s = Settings()
        assert s.api_key_for("_2") == "key-secondary"

    def test_api_key_for_with_suffix_falls_back_to_primary(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY_2", raising=False)
        monkeypatch.setenv("LLM_API_KEY", "key-primary")
        s = Settings()
        assert s.api_key_for("_2") == "key-primary"

    def test_api_key_for_no_env_returns_default(self):
        # _clear_env already erased LLM_API_KEY
        s = Settings()
        assert s.api_key_for() == ""


class TestSettingsBaseUrlFor:
    def test_base_url_for_no_suffix(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "http://custom:8080/v1")
        s = Settings()
        assert s.base_url_for() == "http://custom:8080/v1"

    def test_base_url_for_with_suffix(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL_2", "http://second:8080/v1")
        s = Settings()
        assert s.base_url_for("_2") == "http://second:8080/v1"

    def test_base_url_for_with_suffix_falls_back_to_primary(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL_2", raising=False)
        monkeypatch.setenv("LLM_BASE_URL", "http://fallback:8080/v1")
        s = Settings()
        assert s.base_url_for("_2") == "http://fallback:8080/v1"


class TestGetSettings:
    def test_get_settings_returns_settings_instance(self):
        s = get_settings()
        assert isinstance(s, Settings)

    def test_get_settings_is_cached(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_get_settings_uses_env_vars(self, monkeypatch):
        # get_settings is @lru_cache'd, so clear cache then set env.
        get_settings.cache_clear()
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        s = get_settings()
        assert s.llm_provider == "ollama"


class TestPresets:
    def test_presets_contains_expected_keys(self):
        from llm_lab.settings import PRESETS
        for key in ("cheap", "balanced", "best", "quick"):
            assert key in PRESETS, f"missing preset: {key}"

    def test_resolve_preset_cheap(self):
        from llm_lab.settings import resolve_preset
        cfg = resolve_preset("cheap")
        assert isinstance(cfg, dict)
        assert "model" in cfg

    def test_resolve_preset_balanced(self):
        from llm_lab.settings import resolve_preset
        cfg = resolve_preset("balanced")
        assert "model" in cfg

    def test_resolve_preset_best(self):
        from llm_lab.settings import resolve_preset
        cfg = resolve_preset("best")
        assert "model" in cfg

    def test_resolve_preset_quick(self):
        from llm_lab.settings import resolve_preset
        cfg = resolve_preset("quick")
        assert "model" in cfg

    def test_resolve_preset_case_insensitive(self):
        from llm_lab.settings import resolve_preset
        cfg = resolve_preset("CHEAP")
        assert "model" in cfg

    def test_resolve_preset_none_returns_none(self):
        from llm_lab.settings import resolve_preset
        assert resolve_preset(None) is None

    def test_resolve_preset_unknown_raises(self):
        from llm_lab.settings import resolve_preset
        import pytest
        with pytest.raises(KeyError, match="unknown preset"):
            resolve_preset("nonexistent")

    def test_preset_values_differ(self):
        from llm_lab.settings import PRESETS
        cheap = PRESETS["cheap"]["model"]
        best = PRESETS["best"]["model"]
        assert cheap != best, "cheap and best should use different models"
