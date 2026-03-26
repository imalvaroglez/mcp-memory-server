"""
Local LLM engine for intelligent memory reasoning.

Uses llama-cpp-python to run a quantized LLM (GGUF format) entirely
on-device. The LLM provides reasoning capabilities over stored memories:
- Answering questions about past decisions and patterns
- Analyzing memory sets for patterns, conflicts, and gaps
- Generating intelligent summaries of project context

Default model: Gemma 2 2B Instruct (Q4_K_M GGUF, ~1.5GB)
"""

import json
import logging
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL_REPO = "bartowski/gemma-2-2b-it-GGUF"
DEFAULT_MODEL_FILE = "gemma-2-2b-it-Q4_K_M.gguf"
DEFAULT_MODEL_DIR = Path.home() / ".mcp-memory" / "models"
DEFAULT_CONTEXT_SIZE = 2048
DEFAULT_MAX_TOKENS = 512

# System prompt optimized for coding memory assistant tasks
SYSTEM_PROMPT = """You are a memory reasoning assistant for software developers. \
You help analyze, summarize, and reason about memories stored during coding sessions.

Your role:
- Answer questions about past decisions, patterns, and approaches
- Identify patterns and conflicts across memories
- Provide concise, actionable insights
- Always cite which memories support your reasoning

Format your responses clearly. When referencing memories, use [Memory N] notation.
Be concise and focus on what's actionable for the developer."""


@dataclass
class LLMResult:
    """Result of an LLM inference call."""

    text: str
    tokens_generated: int
    inference_ms: float
    prompt_tokens: int
    model: str


@dataclass
class LLMInfo:
    """Information about the loaded LLM."""

    model_name: str
    model_path: str
    context_size: int
    loaded: bool
    ram_estimate_mb: float
    gpu_layers: int
    platform: str


class LLMEngine:
    """Local LLM engine using llama-cpp-python.

    Lazy-loads the model on first inference call. Automatically downloads
    the GGUF model from HuggingFace if not present locally.

    Usage:
        engine = LLMEngine()
        result = engine.generate("What patterns have I used?")
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_repo: str = DEFAULT_MODEL_REPO,
        model_file: str = DEFAULT_MODEL_FILE,
        context_size: int = DEFAULT_CONTEXT_SIZE,
        n_gpu_layers: int = -1,  # -1 = auto-detect
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """Initialize LLM engine.

        Args:
            model_path: Explicit path to a GGUF model file.
            model_repo: HuggingFace repo ID for auto-download.
            model_file: GGUF filename within the repo.
            context_size: Context window size in tokens.
            n_gpu_layers: Number of layers to offload to GPU (-1 = auto).
            max_tokens: Default max tokens for generation.
        """
        self.model_path = model_path
        self.model_repo = model_repo
        self.model_file = model_file
        self.context_size = context_size
        self.n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens
        self._llm = None
        self._model_name = model_file.replace(".gguf", "")
        self._load_time_ms: float = 0.0

    def _resolve_model_path(self) -> Path:
        """Resolve the model path, downloading if necessary.

        Returns:
            Path to the GGUF model file.
        """
        if self.model_path and Path(self.model_path).exists():
            return Path(self.model_path)

        # Check default model directory
        model_dir = DEFAULT_MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        local_path = model_dir / self.model_file

        if local_path.exists():
            logger.info(f"Using cached model: {local_path}")
            return local_path

        # Download from HuggingFace
        logger.info(
            f"Downloading model {self.model_repo}/{self.model_file} "
            f"(this may take a few minutes on first run)..."
        )
        try:
            from huggingface_hub import hf_hub_download

            downloaded_path = hf_hub_download(
                repo_id=self.model_repo,
                filename=self.model_file,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
            )
            logger.info(f"Model downloaded to: {downloaded_path}")
            return Path(downloaded_path)
        except ImportError:
            raise RuntimeError(
                "huggingface-hub is required for model auto-download. "
                "Install with: poetry install --extras llm"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model: {e}. "
                f"You can manually download from https://huggingface.co/{self.model_repo} "
                f"and place the file at {local_path}"
            )

    def _detect_gpu_layers(self) -> int:
        """Auto-detect optimal GPU layer count.

        Returns:
            Number of layers to offload to GPU.
        """
        if self.n_gpu_layers != -1:
            return self.n_gpu_layers

        system = platform.system()
        machine = platform.machine()

        if system == "Darwin" and machine == "arm64":
            # Apple Silicon: offload all layers to Metal
            logger.info("Detected Apple Silicon — using Metal GPU acceleration")
            return 99  # All layers to GPU

        # Check for CUDA
        try:
            import llama_cpp

            if hasattr(llama_cpp, "llama_supports_gpu_offload"):
                if llama_cpp.llama_supports_gpu_offload():
                    logger.info("CUDA GPU detected — offloading layers")
                    return 35  # Conservative default for CUDA
        except Exception:
            pass

        logger.info("No GPU acceleration detected — running on CPU")
        return 0

    def _load_model(self) -> None:
        """Load the LLM model (lazy, called on first inference)."""
        if self._llm is not None:
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is required for LLM features. "
                "Install with: poetry install --extras llm"
            )

        model_path = self._resolve_model_path()
        gpu_layers = self._detect_gpu_layers()

        logger.info(
            f"Loading LLM: {model_path.name} "
            f"(context={self.context_size}, gpu_layers={gpu_layers})..."
        )

        start = time.time()
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=self.context_size,
            n_gpu_layers=gpu_layers,
            verbose=False,
            # Optimize for memory efficiency
            use_mmap=True,
            use_mlock=False,
        )
        self._load_time_ms = (time.time() - start) * 1000
        logger.info(f"LLM loaded in {self._load_time_ms:.0f}ms")

    def generate(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
        stop: Optional[list[str]] = None,
    ) -> LLMResult:
        """Generate text using the local LLM.

        Args:
            prompt: User prompt text.
            system_prompt: System prompt for context setting.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0=deterministic, 1=creative).
            stop: Stop sequences.

        Returns:
            LLMResult with generated text and metrics.
        """
        self._load_model()

        if max_tokens is None:
            max_tokens = self.max_tokens

        # Format as chat messages for instruction-tuned models
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        start = time.time()
        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
        )
        elapsed = (time.time() - start) * 1000

        text = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})

        return LLMResult(
            text=text.strip(),
            tokens_generated=usage.get("completion_tokens", 0),
            inference_ms=round(elapsed, 2),
            prompt_tokens=usage.get("prompt_tokens", 0),
            model=self._model_name,
        )

    def analyze_memories(
        self,
        memories: list[dict],
        query: str,
        project: Optional[str] = None,
    ) -> LLMResult:
        """Analyze a set of memories to answer a question.

        Formats memories into a structured prompt and asks the LLM
        to reason about them.

        Args:
            memories: List of memory dicts with 'text', 'metadata', etc.
            query: The question to answer.
            project: Optional project context.

        Returns:
            LLMResult with the analysis.
        """
        # Format memories into context
        memory_context = self._format_memories(memories)

        prompt = f"""Based on the following stored memories, answer this question:

**Question:** {query}
{f"**Project:** {project}" if project else ""}

**Stored Memories:**
{memory_context}

Provide a clear, concise answer. Reference specific memories using [Memory N] notation. \
If the memories don't contain enough information, say so."""

        return self.generate(prompt, max_tokens=self.max_tokens)

    def detect_patterns(
        self,
        memories: list[dict],
        focus: Optional[str] = None,
    ) -> LLMResult:
        """Detect patterns, conflicts, and gaps across memories.

        Args:
            memories: List of memory dicts.
            focus: Optional focus area for analysis.

        Returns:
            LLMResult with structured analysis.
        """
        memory_context = self._format_memories(memories)

        prompt = f"""Analyze the following stored memories for a software project. \
{f"Focus on: {focus}" if focus else ""}

**Stored Memories:**
{memory_context}

Provide analysis in this structure:

**Patterns Found:**
- List recurring patterns, conventions, and approaches

**Potential Conflicts:**
- List any contradicting decisions or approaches

**Knowledge Gaps:**
- List areas where important context is missing

**Recommendations:**
- Suggest what the developer should remember or clarify next

Be concise and actionable."""

        return self.generate(prompt, max_tokens=self.max_tokens)

    def summarize_context(
        self,
        memories: list[dict],
        project: Optional[str] = None,
    ) -> LLMResult:
        """Generate an intelligent summary of project memories.

        Args:
            memories: List of memory dicts.
            project: Optional project name.

        Returns:
            LLMResult with the summary.
        """
        memory_context = self._format_memories(memories)

        prompt = f"""Summarize the following project memories into a concise context briefing.
{f"**Project:** {project}" if project else ""}

**Stored Memories:**
{memory_context}

Create a developer briefing that covers:
1. Key architectural decisions
2. Established patterns and conventions
3. Important gotchas and lessons learned
4. Current state and next steps

Keep it concise — this will be read at the start of a coding session."""

        return self.generate(prompt, max_tokens=self.max_tokens)

    def _format_memories(self, memories: list[dict]) -> str:
        """Format memory dicts into a numbered text context.

        Args:
            memories: List of memory dicts.

        Returns:
            Formatted string with numbered memories.
        """
        parts = []
        for i, mem in enumerate(memories, 1):
            text = mem.get("text", "")[:300]  # Truncate long memories
            metadata = mem.get("metadata", {})
            mem_type = metadata.get("type", "note")
            importance = metadata.get("importance", "—")
            created = mem.get("created_at", "unknown")

            parts.append(
                f"[Memory {i}] ({mem_type}, importance: {importance}, {created})\n{text}"
            )
        return "\n\n".join(parts)

    def info(self) -> LLMInfo:
        """Get information about the LLM engine.

        Returns:
            LLMInfo with model details.
        """
        model_path = ""
        try:
            resolved = self._resolve_model_path()
            model_path = str(resolved)
            size_mb = resolved.stat().st_size / (1024 * 1024) if resolved.exists() else 0
        except Exception:
            size_mb = 0

        return LLMInfo(
            model_name=self._model_name,
            model_path=model_path,
            context_size=self.context_size,
            loaded=self._llm is not None,
            ram_estimate_mb=round(size_mb * 1.2, 1),  # ~1.2x model size in RAM
            gpu_layers=self._detect_gpu_layers(),
            platform=f"{platform.system()} {platform.machine()}",
        )

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            logger.info("LLM model unloaded")


# Module-level engine instance (lazy)
_engine: Optional[LLMEngine] = None


def get_engine(
    model_path: Optional[Path] = None,
    model_repo: str = DEFAULT_MODEL_REPO,
    model_file: str = DEFAULT_MODEL_FILE,
    context_size: int = DEFAULT_CONTEXT_SIZE,
) -> LLMEngine:
    """Get or create the module-level LLM engine instance."""
    global _engine
    if _engine is None:
        _engine = LLMEngine(
            model_path=model_path,
            model_repo=model_repo,
            model_file=model_file,
            context_size=context_size,
        )
    return _engine
