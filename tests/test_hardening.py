"""Regression tests for production hardening behavior."""

import json
import importlib
from pathlib import Path
from unittest.mock import patch

from semantic_code_intel.api.schemas import SearchRequest
from semantic_code_intel.config import CodeIntelConfig, GenerationConfig
from semantic_code_intel.generation.synthesizer import CodeSynthesizer
from semantic_code_intel.parser.base import CodeChunk, SymbolType
from semantic_code_intel.retrieval.citation import SearchResult

api_module = importlib.import_module("semantic_code_intel.api.app")


def _result() -> SearchResult:
    chunk = CodeChunk(
        chunk_id="answer-1",
        file_path="src/auth.py",
        absolute_path="/repo/src/auth.py",
        language="python",
        symbol_name="authenticate",
        symbol_type=SymbolType.FUNCTION,
        start_line=10,
        end_line=12,
        content="def authenticate(token):\n    return verify(token)",
    )
    return SearchResult.from_chunk(chunk, score=0.9)


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps({"response": "Authentication delegates to verification [src/auth.py:L10-L12]."}).encode()


def test_ollama_synthesis_uses_generated_response():
    config = CodeIntelConfig(generation=GenerationConfig(provider="ollama", fallback_to_extractive=False))
    with patch("semantic_code_intel.generation.synthesizer.urlopen", return_value=_Response()):
        response = CodeSynthesizer(config).synthesize("How is auth checked?", [_result()])
    assert response.provider == "ollama"
    assert "delegates to verification" in response.answer

    with patch("semantic_code_intel.generation.synthesizer.urlopen", return_value=_Response()) as mocked:
        CodeSynthesizer(config).synthesize("How is auth checked?", [_result()])
    sent_prompt = json.loads(mocked.call_args.args[0].data)["prompt"]
    assert "## Direct answer" in sent_prompt
    assert "Execution walkthrough" in sent_prompt
    assert "src/auth.py:L10-L12" in sent_prompt


def test_ollama_failure_is_explicitly_labeled_fallback():
    config = CodeIntelConfig(generation=GenerationConfig(provider="ollama", fallback_to_extractive=True))
    with patch("semantic_code_intel.generation.synthesizer.urlopen", side_effect=TimeoutError()):
        response = CodeSynthesizer(config).synthesize("How is auth checked?", [_result()])
    assert response.provider == "extractive-fallback"
    assert "src/auth.py" in response.answer
    assert "## Source-backed walkthrough" in response.answer
    assert "## Evidence gaps" in response.answer


def test_api_uses_canonical_schema_and_bounded_cache(tmp_path: Path):
    assert api_module.SearchRequest is SearchRequest
    old_size = api_module._PIPELINE_CACHE_SIZE
    old_config = api_module.config
    old_pipeline = api_module.pipeline
    old_repo = api_module._ACTIVE_REPO_PATH
    old_index = api_module._ACTIVE_INDEX_PATH
    try:
        api_module._PIPELINE_CACHE_SIZE = 2
        api_module._PIPELINES.clear()
        for name in ("one", "two", "three"):
            repo = tmp_path / name
            repo.mkdir()
            api_module.get_pipeline(str(repo))
        assert len(api_module._PIPELINES) == 2
    finally:
        api_module._PIPELINES.clear()
        api_module._PIPELINE_CACHE_SIZE = old_size
        api_module.config = old_config
        api_module.pipeline = old_pipeline
        api_module._ACTIVE_REPO_PATH = old_repo
        api_module._ACTIVE_INDEX_PATH = old_index
