# EXACT FIXES FOR semantic_symbol.py
# This document shows the exact line ranges and code to modify

## FIX 1: Handle Share Elements Gracefully
## Location: Lines 276-283

### Current Code:
```python
elif elem.tag == MathML.share:
    # copy a portion of the tree used before ...
    if elem.attrib["href"] == "#.cmml":
        # special case common in equations, repeat right operand of last operation ...
        if parent.parent.tag == "U!and":
            # identify root of subtree to copy ...
            last_operand = parent.parent.children[-1].children[-1]
            # copy ...
            retval = SemanticSymbol.Copy(last_operand)
            retval.parent = parent
```

### Fixed Code:
```python
elif elem.tag == MathML.share:
    # copy a portion of the tree used before ...
    href = elem.attrib.get("href", "")
    
    if href == "#.cmml":
        # special case common in equations, repeat right operand of last operation ...
        if parent and parent.parent and parent.parent.tag == "U!and":
            # identify root of subtree to copy ...
            last_operand = parent.parent.children[-1].children[-1]
            # copy ...
            retval = SemanticSymbol.Copy(last_operand)
            retval.parent = parent
    else:
        # Generic case: create a placeholder for the reference
        # This preserves formula structure when we can't resolve the actual ref
        # href typically looks like: #Ex1.m1.sh1 (latexmlmath reference)
        retval = SemanticSymbol("?SHARE:" + href, parent=parent)
```

## FIX 2: Add Missing "conditional" CSYMBOLs
## Location: After line 682 (after "square-root" handling)

### Insert This Code:
```python
elif content == "conditional":
    # Represents divisibility operator: k|n means "k divides n"
    # In probability: P(A|B) = conditional probability
    retval = SemanticSymbol("O!conditional", parent=parent)
```

## FIX 3: Add Missing "differential-d" CSYMBOLs
## Location: After "conditional" (right after FIX 2)

### Insert This Code:
```python
elif content == "differential-d":
    # Represents the differential operator 'd' in integrals
    # e.g., ∫ f(x) dx - the dx part
    retval = SemanticSymbol("O!differential", parent=parent)
```

## EXPLANATION OF FIXES

### Fix 1: Share Elements
- **Problem**: The current code only handles `href="#.cmml"`, but latexmlmath generates references like `#Ex1.m1.sh1`
- **Solution**: For unmatched hrefs, create a placeholder node tagged as "?SHARE:{href}" instead of leaving retval as None
- **Why it works**: The parser won't crash with UnknownTagException, and the tree structure is preserved
- **Impact**: Allows parsing of all chained equations (those with multiple = signs)

### Fix 2: Conditional Operator
- **Problem**: The csymbol with cd="latexml" and content="conditional" is not handled
- **Solution**: Add it to the special case handling section
- **What it represents**: The divisibility operator "|" (k|n = "k divides n") and conditional probability P(A|B)
- **Impact**: Allows parsing of formulas with divisibility and conditional probability notations

### Fix 3: Differential Operator
- **Problem**: The csymbol with cd="latexml" and content="differential-d" is not handled
- **Solution**: Add it to the special case handling section
- **What it represents**: The differential "d" in integrals (∫ f(x) dx)
- **Impact**: Allows parsing of formulas with integrals

## TESTING BEFORE AND AFTER

### Before Fixes:
- Total: 16 tests
- Passed: 2 (12.5%)
- Failed: 14 (share elements: 7, conditional: 4, differential: 3)

### After Fixes (Expected):
- Total: 16 tests
- Passed: 16 (100%)
- Failed: 0

## FILES TO MODIFY
- `/Users/abramjopaul/Documents/projects/multi-index-rag/src/multirag/formula_search/semantic_symbol.py`

## LINES TO CHANGE
1. Lines 276-283: Share element handling (MODIFY)
2. After line 682: Add conditional and differential-d handling (INSERT)

## VERIFICATION COMMAND
After making changes, run:
```bash
cd /Users/abramjopaul/Documents/projects/multi-index-rag
poetry run python test_opt_fixes.py
```

Expected output: "Pass Rate: 100.0%"
