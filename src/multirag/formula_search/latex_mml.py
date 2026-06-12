"""
LaTeX to MathML conversion using latexmlmath.

Based on TangentCFT (Tangent Combined FastText)
Original repository: https://github.com/BehroozMansouri/TangentCFT

Original authors: Nidhin Pattaniyil, Frank Wm. Tompa
Adapted for multirag formula_search module with fallback stylesheet resolution
"""

import multiprocessing
import os
import platform
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

__author__ = "Nidhin, FWTompa"

# Module-level caching to avoid repeated filesystem checks and platform detection
_STYLESHEET_CACHE: Optional[str] = None
_PLATFORM_SYSTEM: str = platform.system()
_OPTIMAL_WORKERS_CACHE: Optional[int] = None


def _stylesheet_path() -> str:
    return os.path.join(os.path.dirname(__file__), "mws.sty.ltxml")


def _require_stylesheet() -> str:
    """Get and validate stylesheet path, caching result for repeated use."""
    global _STYLESHEET_CACHE
    
    if _STYLESHEET_CACHE is not None:
        return _STYLESHEET_CACHE
    
    qvar_template_file = _stylesheet_path()
    if not os.path.exists(qvar_template_file):
        error_msg = (
            f"Stylesheet for MathWeb Search (mws.sty.ltxml) not found at:\n"
            f"  {qvar_template_file}\n"
            f"This file is required for LaTeX to MathML conversion."
        )
        sys.exit(error_msg)
    
    _STYLESHEET_CACHE = qvar_template_file
    return qvar_template_file


def _get_optimal_workers() -> int:
    """
    Calculate optimal number of worker processes.
    
    Heuristic: Reserve 1 core for host process, cap at 8 unless on high-core-count systems.
    On Linux VMs with many cores, scales up gracefully. On macOS/typical systems, stays conservative.
    
    Returns:
        int: Optimal worker count (minimum 1)
    """
    global _OPTIMAL_WORKERS_CACHE
    
    if _OPTIMAL_WORKERS_CACHE is not None:
        return _OPTIMAL_WORKERS_CACHE
    
    cpu_count = multiprocessing.cpu_count()
    # Reserve 1 core for host process; cap at 8 for typical systems, allow more on high-core systems
    optimal = max(1, min(cpu_count - 1, 8))
    _OPTIMAL_WORKERS_CACHE = optimal
    return optimal


def _normalize_tex_query(tex_query: str) -> str:
    """Remove stray percent signs not escaped with backslash."""
    return re.sub(r"([^\\])%", r"\1", tex_query)


def _substitute_qvar_tags(result: str) -> str:
    """Replace generic mi tags with MathWeb Search qvar tags."""
    return re.sub(
        r"<mi.*?>qvar_(.*)</mi>",
        r'<mws:qvar xmlns:mws="http://search.mathweb.org/ns" name="\1"/>',
        result,
    )


def _convert_with_latexmlmath(tex_query: str, output_flag: str) -> str:
    """
    Convert LaTeX to MathML using latexmlmath subprocess.
    
    Args:
        tex_query: LaTeX formula string
        output_flag: 'pmml' or 'cmml' for output format
        
    Returns:
        MathML string with qvar tags substituted
        
    Raises:
        Exception: If latexmlmath subprocess fails
    """
    # Stylesheet validation is cached; this just retrieves it
    qvar_template_file = _require_stylesheet()
    tex_query = _normalize_tex_query(tex_query)

    use_shell = "Windows" in _PLATFORM_SYSTEM
    p2 = subprocess.Popen(
        [
            "latexmlmath",
            f"--{output_flag}=-",
            "--preload=amsmath",
            "--preload=amsfonts",
            "--preload=" + qvar_template_file,
            "-",
        ],
        shell=use_shell,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output, err = p2.communicate(input=tex_query.encode())

    if (not output) and err:
        print("Error in converting LaTeX to MathML: " + tex_query, file=sys.stderr)
        raise Exception(str(err))

    try:
        result = output.decode("utf-8")
        result = _substitute_qvar_tags(result)
    except UnicodeDecodeError as uae:
        print("Failed to decode " + uae.reason, file=sys.stderr)
        result = output.decode("utf-8", "replace")
        print("Decoded %s" % result)
    except:
        print("Failure in converting LaTeX in " + tex_query, file=sys.stderr)
        raise

    return result


def _pool_convert_worker(tex_query: str, mathml_type: str) -> str:
    """Worker function for ProcessPoolExecutor (must be module-level for pickling on macOS)."""
    return _convert_with_latexmlmath(tex_query, mathml_type)


class LatexToMathMLPool:
    """
    Context-managed subprocess pool for batch LaTeX-to-MathML conversion.
    
    Persistent pool keeps latexmlmath worker processes alive across multiple batches,
    avoiding subprocess spawn overhead (~100ms per formula). Enables conversion of
    large formula corpora at 500-1000 formulas/sec instead of 9-10 formulas/sec.
    
    Usage:
        with LatexToMathMLPool(num_workers=4) as pool:
            results = pool.convert_batch(["x^2", "a+b", ...])
            
    Or via the LatexToMathML.convert_batch2() class method (which manages pool lifecycle).
    """
    
    def __init__(self, num_workers: Optional[int] = None, mathml_type: str = "cmml"):
        """
        Initialize conversion pool.
        
        Args:
            num_workers: Number of worker processes. If None, auto-tunes based on CPU count.
                        Heuristic: reserves 1 core for host, caps at 8 for typical systems.
            mathml_type: 'pmml' (Presentation) or 'cmml' (Content) output format.
            
        Raises:
            ValueError: If num_workers < 1 or mathml_type is invalid
        """
        if num_workers is None:
            num_workers = _get_optimal_workers()

        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")

        if mathml_type not in {"pmml", "cmml"}:
            raise ValueError("mathml_type must be 'pmml' or 'cmml'")

        self.num_workers = num_workers
        self.mathml_type = mathml_type
        self._stylesheet = _require_stylesheet()  # Validate before creating executor
        self._executor = ProcessPoolExecutor(max_workers=self.num_workers)

    def convert_batch(self, tex_queries: List[str]) -> List[Optional[str]]:
        """
        Convert batch of LaTeX formulas to MathML using worker pool.
        
        Preserves order and handles per-formula failures gracefully.
        
        Args:
            tex_queries: List of LaTeX strings
            
        Returns:
            List of MathML strings (same order as input) or None for failures
        """
        if not tex_queries:
            return []

        results: List[Optional[str]] = [None] * len(tex_queries)
        future_to_idx = {
            self._executor.submit(_pool_convert_worker, tex, self.mathml_type): idx
            for idx, tex in enumerate(tex_queries)
        }

        for future in future_to_idx:
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(
                    f"Error converting formula {idx}: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                results[idx] = None

        return results

    def shutdown(self) -> None:
        """Cleanly shut down worker processes."""
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False


class LatexToMathML(object):
    """
    Backward-compatible API for LaTeX-to-MathML conversion.
    
    Provides both single-formula and batch conversion methods.
    Single-formula methods spawn fresh subprocesses and should only be used for
    occasional conversions; for large batches, use convert_batch2() instead.
    """
    
    @classmethod
    def convert_to_mathml(cls, tex_query):
        """
        Convert single LaTeX formula to Presentation MathML (PMML).
        
        Warning: Spawns a new subprocess per call. Use convert_batch2() for bulk conversion.
        
        Args:
            tex_query: LaTeX formula string
            
        Returns:
            Presentation MathML string or None if conversion fails
        """
        return _convert_with_latexmlmath(tex_query, "pmml")

    @classmethod
    def convert_to_mathml2(cls, tex_query):
        """
        Convert single LaTeX formula to Content MathML (CMML).
        
        Warning: Spawns a new subprocess per call. Use convert_batch2() for bulk conversion.
        
        Args:
            tex_query: LaTeX formula string
            
        Returns:
            Content MathML string with qvar tags substituted, or None if conversion fails
        """
        result = _convert_with_latexmlmath(tex_query, "cmml")
        if r"<mi.*?>qvar_(.*)</mi>" in result:
            print("Contains qvar\n")
            print(result)
        return result

    @classmethod
    def convert_batch2(
        cls, tex_queries: List[str], num_workers: Optional[int] = None
    ) -> List[Optional[str]]:
        """
        Convert multiple LaTeX formulas to Content MathML (CMML) using a persistent subprocess pool.

        Uses --cmml=- output format (Content MathML).
        The pool keeps latexmlmath worker processes alive across calls so batch conversion
        avoids spawning a new subprocess for every formula.

        Args:
            tex_queries: List of LaTeX strings to convert
            num_workers: Number of worker processes. If None, auto-tunes based on CPU count
                        (reserves 1 core for host, caps at 8 for typical systems).
                        On high-core-count systems (e.g., Linux VMs), scales up gracefully.

        Returns:
            List of MathML strings (content format) in same order as input, or None for failures

        Example:
            >>> formulas = ["x^2 + y", "a + b"]
            >>> results = LatexToMathML.convert_batch2(formulas)  # Auto-tunes workers
            >>> results = LatexToMathML.convert_batch2(formulas, num_workers=4)  # Override
        """
        with LatexToMathMLPool(num_workers=num_workers, mathml_type="cmml") as pool:
            return pool.convert_batch(tex_queries)
