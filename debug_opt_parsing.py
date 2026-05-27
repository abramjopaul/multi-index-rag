"""
Debug script to identify OPT generation failures and analyze MathML structures.

This script:
1. Tests formulas that are known to fail
2. Generates MathML for each formula
3. Analyzes the MathML structure
4. Attempts to parse it with the current SemanticSymbol parser
5. Identifies problematic MathML elements
"""

import logging
import sys
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from collections import defaultdict
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Configure logging to see errors
logging.basicConfig(
    level=logging.ERROR,  # Changed to ERROR to reduce noise
    format='%(name)s - %(levelname)s - %(message)s'
)

from multirag.formula_search.latex_mml import LatexToMathML
from multirag.formula_search.math_extractor import MathExtractor
from multirag.formula_search.exceptions import UnknownTagException
from multirag.formula_search.mathml import MathML

# List of failing formulas from the terminal output
FAILING_FORMULAS = [
    r"1.000\ldots -.99999\ldots = .000\ldots = 0",
    r"(a \times b)\cdot a = a_1(a_2b_3-a_3b_2)-a_2(a_1b_3-a_3b_1)-a_3(a_1b_2-a_2b_1)=0",
    r"0.9999... < x < 1",
    r"\langle \, f, \, g \, \rangle = \int_a^b \ f(x)\overline{g(x)} \, dx",
    r"a = CE = L/R=2\sin(\theta)",
    r"\frac{\frac{9}{10}}{1 - \frac{1}{10}} = \frac{\frac{9}{10}}{\frac{9}{10}} = 1.",
    r"a^2 / a^3 = a^{-1} = 1/a",
    r"k|n",
    r"p|(10^k-1)",
    r"k|(p-1)",
    r"P(X_n | X_1, X_2, \dots X_{n-1}) = P(X_n | X_{n-1})",
]

# List of working formulas (for comparison)
WORKING_FORMULAS = [
    r"R^2 = 2 = \frac{Q^2}{D^2}",
    r"2^{1-1} = 2^1 / 2^1 = 2/2 = 1",
    r"x^2 + y",
]


class MathMLAnalyzer:
    """Analyze MathML structures and identify problematic elements."""
    
    def __init__(self):
        self.problematic_elements = defaultdict(int)
        self.problematic_csymbols = defaultdict(int)
        self.share_hrefs = defaultdict(int)
        
    def analyze_mathml(self, mathml_str: str) -> Dict:
        """Parse and analyze MathML structure."""
        try:
            # Parse the MathML
            root = ET.fromstring(mathml_str)
            
            # Get namespace
            ns = {'m': 'http://www.w3.org/1998/Math/MathML'}
            
            analysis = {
                'valid': True,
                'elements': self._count_elements(root, ns),
                'csymbols': self._find_csymbols(root, ns),
                'shares': self._find_shares(root, ns),
                'errors': self._find_errors(root, ns),
                'raw_root_tag': root.tag,
            }
            
            return analysis
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
            }
    
    def _count_elements(self, elem, ns, counts=None):
        """Count occurrences of each element type."""
        if counts is None:
            counts = defaultdict(int)
        
        # Remove namespace for display
        tag = elem.tag.replace('{http://www.w3.org/1998/Math/MathML}', '')
        counts[tag] += 1
        
        for child in elem:
            self._count_elements(child, ns, counts)
        
        return dict(counts)
    
    def _find_csymbols(self, elem, ns):
        """Find all csymbol elements and their attributes."""
        csymbols = []
        
        for child in elem.iter('{http://www.w3.org/1998/Math/MathML}csymbol'):
            cd = child.attrib.get('cd', 'NO_CD')
            text = (child.text or '').strip()
            csymbols.append({
                'cd': cd,
                'content': text,
                'attribs': dict(child.attrib),
            })
            self.problematic_csymbols[f"{cd}:{text}"] += 1
        
        return csymbols
    
    def _find_shares(self, elem, ns):
        """Find all share elements and their hrefs."""
        shares = []
        
        for child in elem.iter('{http://www.w3.org/1998/Math/MathML}share'):
            href = child.attrib.get('href', 'NO_HREF')
            shares.append({'href': href, 'attribs': dict(child.attrib)})
            self.share_hrefs[href] += 1
        
        return shares
    
    def _find_errors(self, elem, ns):
        """Find merror elements."""
        errors = []
        
        for child in elem.iter('{http://www.w3.org/1998/Math/MathML}merror'):
            errors.append({'tag': child.tag, 'text': (child.text or '').strip()})
        
        return errors


def test_formula(formula: str, label: str = "Test") -> Dict:
    """Test a single formula and return analysis."""
    print(f"\n{'='*80}")
    print(f"{label}: {formula}")
    print('='*80)
    
    result = {
        'formula': formula,
        'label': label,
        'mathml_generated': False,
        'cmml_isolated': False,
        'parsing_success': False,
        'error': None,
        'analysis': None,
    }
    
    try:
        # Step 1: Generate MathML
        print("Step 1: Converting LaTeX to MathML...")
        mathml = LatexToMathML.convert_to_mathml2(formula)
        result['mathml_generated'] = True
        print("✓ MathML generated successfully")
        print(f"  MathML (first 200 chars): {mathml[:200]}...")
        
        # Step 2: Isolate CMML
        print("\nStep 2: Isolating Content MathML...")
        cmml = MathExtractor.isolate_cmml(mathml)
        result['cmml_isolated'] = True
        print("✓ Content MathML isolated successfully")
        print(f"  CMML (first 300 chars): {cmml[:300]}...")
        
        # Step 3: Analyze CMML
        print("\nStep 3: Analyzing CMML structure...")
        analyzer = MathMLAnalyzer()
        analysis = analyzer.analyze_mathml(cmml)
        result['analysis'] = analysis
        
        if analysis['valid']:
            print("✓ CMML analysis successful")
            print(f"  Elements: {analysis['elements']}")
            if analysis['csymbols']:
                print(f"  CSYMBOLs: {analysis['csymbols']}")
            if analysis['shares']:
                print(f"  SHAREs: {analysis['shares']}")
            if analysis['errors']:
                print(f"  MERRORs: {analysis['errors']}")
        else:
            print(f"✗ CMML analysis failed: {analysis.get('error')}")
            result['error'] = analysis.get('error')
        
        # Step 4: Try to parse
        print("\nStep 4: Attempting to parse CMML with SemanticSymbol...")
        symbol_root = MathExtractor.convert_to_semanticsymbol(cmml)
        result['parsing_success'] = symbol_root is not None
        
        if symbol_root:
            print(f"✓ Parsing successful! Root tag: {symbol_root.tag}")
        else:
            print("✗ Parsing returned None")
            result['error'] = "Parsing returned None"
        
    except UnknownTagException as e:
        print(f"✗ UnknownTagException: {e}")
        result['error'] = f"UnknownTagException: {e}"
    except Exception as e:
        print(f"✗ Exception: {type(e).__name__}: {e}")
        result['error'] = f"{type(e).__name__}: {e}"
    
    return result


def main():
    """Run all tests and generate report."""
    print("\n" + "="*80)
    print("OPT GENERATION DEBUG REPORT")
    print("="*80)
    
    analyzer = MathMLAnalyzer()
    all_results = []
    
    # Test working formulas first
    print("\n\n" + "="*80)
    print("TESTING WORKING FORMULAS (baseline)")
    print("="*80)
    
    for i, formula in enumerate(WORKING_FORMULAS, 1):
        result = test_formula(formula, f"WORKING-{i}")
        all_results.append(result)
    
    # Test failing formulas
    print("\n\n" + "="*80)
    print("TESTING FAILING FORMULAS")
    print("="*80)
    
    for i, formula in enumerate(FAILING_FORMULAS, 1):
        result = test_formula(formula, f"FAILING-{i}")
        all_results.append(result)
    
    # Generate summary report
    print("\n\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    passed = sum(1 for r in all_results if r['parsing_success'])
    failed = sum(1 for r in all_results if not r['parsing_success'])
    
    print(f"\nTotal formulas tested: {len(all_results)}")
    print(f"Successful parses: {passed}")
    print(f"Failed parses: {failed}")
    
    print("\n" + "-"*80)
    print("PROBLEMATIC CSYMBOLS (from failing formulas)")
    print("-"*80)
    
    failing_csymbols = defaultdict(int)
    for result in all_results:
        if not result['parsing_success'] and result['analysis'] and result['analysis'].get('valid'):
            for csymbol in result['analysis'].get('csymbols', []):
                cd = csymbol['cd']
                content = csymbol['content']
                key = f"{cd}:{content}"
                failing_csymbols[key] += 1
    
    if failing_csymbols:
        for key, count in sorted(failing_csymbols.items(), key=lambda x: -x[1]):
            print(f"  {key}: {count} occurrences")
    else:
        print("  (No problematic csymbols found)")
    
    print("\n" + "-"*80)
    print("PROBLEMATIC SHARE HREFS (from failing formulas)")
    print("-"*80)
    
    failing_shares = defaultdict(int)
    for result in all_results:
        if not result['parsing_success'] and result['analysis'] and result['analysis'].get('valid'):
            for share in result['analysis'].get('shares', []):
                href = share['href']
                failing_shares[href] += 1
    
    if failing_shares:
        for href, count in sorted(failing_shares.items(), key=lambda x: -x[1]):
            print(f"  {href}: {count} occurrences")
    else:
        print("  (No problematic shares found)")
    
    print("\n" + "-"*80)
    print("ERROR MESSAGES")
    print("-"*80)
    
    error_types = defaultdict(int)
    for result in all_results:
        if result['error']:
            error_types[result['error']] += 1
    
    if error_types:
        for error, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {error}: {count} occurrences")
    else:
        print("  (No errors)")
    
    print("\n" + "-"*80)
    print("FAILED FORMULAS")
    print("-"*80)
    
    for result in all_results:
        if not result['parsing_success']:
            print(f"\n{result['label']}: {result['formula']}")
            print(f"  Error: {result['error']}")
            if result['analysis']:
                print(f"  MathML valid: {result['analysis'].get('valid')}")
    
    print("\n" + "="*80)
    print("END OF REPORT")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
