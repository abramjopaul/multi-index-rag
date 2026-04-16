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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

__author__ = "Nidhin, FWTompa"


class LatexToMathML(object):
    @classmethod
    def convert_to_mathml(cls, tex_query):
        # print("Convert LaTeX to MathML:$"+tex_query+"$",flush=True)
        # Look for mws.sty.ltxml in the current directory or fallback to TangentCFT
        qvar_template_file = os.path.join(os.path.dirname(__file__), "mws.sty.ltxml")
        if not os.path.exists(qvar_template_file):
            # Fallback to external/TangentCFT if not found
            qvar_template_file = os.path.join(
                os.path.dirname(__file__),
                "../../../external/TangentCFT/TangentS/math_tan/mws.sty.ltxml",
            )

        if not os.path.exists(qvar_template_file):
            print("Tried %s" % qvar_template_file, end=": ")
            sys.exit("Stylesheet for wildcard is missing")

        # Make sure there are no isolated % signs in tex_query (introduced by latexmlmath, for example, in 13C.mml test file) (FWT)
        tex_query = re.sub(
            r"([^\\])%", r"\1", tex_query
        )  # remove % not preceded by backslashes (FWT)

        use_shell = "Windows" in platform.system()
        p2 = subprocess.Popen(
            [
                "latexmlmath",
                "--pmml=-",
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
            # strangely, not getting expected conversion. Instead      (FWT)
            #    <mi mathcolor="red" mathvariant="italic">qvar_B</mi>
            # should have been
            #    <mws:qvar xmlns:mws="http://search.mathweb.org/ns" name="B"/>

            result = re.sub(
                r"<mi.*?>qvar_(.*)</mi>",
                r'<mws:qvar xmlns:mws="http://search.mathweb.org/ns" name="\1"/>',
                result,
            )  # FWT
        except UnicodeDecodeError as uae:
            print("Failed to decode " + uae.reason, file=sys.stderr)
            result = output.decode("utf-8", "replace")
            print("Decoded %s" % result)
        except:
            print("Failure in converting LaTeX in " + tex_query, file=sys.stderr)
            raise  # pass on the exception to identify context
        return result

    @classmethod
    def convert_to_mathml2(cls, tex_query):
        # print("Convert LaTeX to MathML:$"+tex_query+"$",flush=True)
        # Look for mws.sty.ltxml in the current directory or fallback to TangentCFT
        qvar_template_file = os.path.join(os.path.dirname(__file__), "mws.sty.ltxml")
        if not os.path.exists(qvar_template_file):
            # Fallback to external/TangentCFT if not found
            qvar_template_file = os.path.join(
                os.path.dirname(__file__),
                "../../../external/TangentCFT/TangentS/math_tan/mws.sty.ltxml",
            )

        if not os.path.exists(qvar_template_file):
            print("Tried %s" % qvar_template_file, end=": ")
            sys.exit("Stylesheet for wildcard is missing")

        # Make sure there are no isolated % signs in tex_query (introduced by latexmlmath, for example, in 13C.mml test file) (FWT)
        tex_query = re.sub(
            r"([^\\])%", r"\1", tex_query
        )  # remove % not preceded by backslashes (FWT)

        use_shell = "Windows" in platform.system()
        p2 = subprocess.Popen(
            [
                "latexmlmath",
                "--cmml=-",
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
            # strangely, not getting expected conversion. Instead      (FWT)
            #    <mi mathcolor="red" mathvariant="italic">qvar_B</mi>
            # should have been
            #    <mws:qvar xmlns:mws="http://search.mathweb.org/ns" name="B"/>

            if r"<mi.*?>qvar_(.*)</mi>" in result:
                print("Contains qvar\n")
                print(result)

            result = re.sub(
                r"<mi.*?>qvar_(.*)</mi>",
                r'<mws:qvar xmlns:mws="http://search.mathweb.org/ns" name="\1"/>',
                result,
            )  # FWT

        except UnicodeDecodeError as uae:
            print("Failed to decode " + uae.reason, file=sys.stderr)
            result = output.decode("utf-8", "replace")
            print("Decoded %s" % result)
        except:
            print("Failure in converting LaTeX in " + tex_query, file=sys.stderr)
            raise  # pass on the exception to identify context
        return result

    @classmethod
    def convert_batch2(
        cls, tex_queries: List[str], num_workers: Optional[int] = None
    ) -> List[str]:
        """
        Convert multiple LaTeX formulas to MathML (Content format) in parallel using thread pool.

        Uses --cmml=- output format (Content MathML).
        ThreadPoolExecutor is used instead of ProcessPoolExecutor because this is I/O-bound
        (subprocess calls) and threads avoid multiprocessing bootstrapping issues on macOS.

        Args:
            tex_queries: List of LaTeX strings to convert
            num_workers: Number of worker threads. If None, auto-tunes based on CPU count (capped at 8)

        Returns:
            List of MathML strings (content format) in same order as input

        Example:
            >>> formulas = ["x^2 + y", "a + b"]
            >>> results = LatexToMathML.convert_batch2(formulas, num_workers=4)
        """
        if not tex_queries:
            return []

        # Auto-tune worker count based on CPU count
        if num_workers is None:
            cpu_count = multiprocessing.cpu_count()
            num_workers = min(cpu_count, 8)  # Cap at 8 threads for I/O-bound operations

        results = [None] * len(tex_queries)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(cls.convert_to_mathml2, tex): idx
                for idx, tex in enumerate(tex_queries)
            }

            # Collect results as they complete
            for future in as_completed(future_to_idx):
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
