"""
Detailed test file showing the exact nature of each parsing failure.

This file demonstrates:
1. Exactly which MathML structures cause failures
2. What the current parser is trying to do
3. What the expected behavior should be
"""

# Issue 1: Share Elements with Different HREFs
# Current Code (lines 276-283):
#   elif elem.tag == MathML.share:
#       if elem.attrib["href"] == "#.cmml":
#           # handles only this specific href
#           if parent.parent.tag == "U!and":
#               # ... copy logic

# Problem: When href is "#Ex1.m1.sh1", the code doesn't set retval,
#          causing UnknownTagException at the end of parse_from_mathml()

# Example MathML for: R^2 = 2 = \frac{Q^2}{D^2}
share_example_1 = """
<share href="#Ex1.m1.sh1" />
"""

# The href "#Ex1.m1.sh1" means "copy the subtree with id Ex1.m1.sh1"
# This is LaTeXML's way of representing repeated elements in chained equations

# Solution: Return a wildcard/placeholder node
# Code should be something like:
fix_1_code = """
elif elem.tag == MathML.share:
    # Handle share elements - references to previously parsed subexpressions
    href = elem.attrib.get("href", "")
    
    if href == "#.cmml":
        # Special case: repeat right operand of last operation
        if parent and parent.parent and parent.parent.tag == "U!and":
            last_operand = parent.parent.children[-1].children[-1]
            retval = SemanticSymbol.Copy(last_operand)
            retval.parent = parent
    else:
        # Generic case: create a placeholder that represents the reference
        # This preserves the formula structure even though we can't resolve the actual ref
        # We use a wildcard identifier: "?SHARE" 
        retval = SemanticSymbol("?SHARE:" + href, parent=parent)
"""

# Issue 2: Missing "conditional" csymbol
# The divisibility operator: k|n means "k divides n"

# Current code has a long list of handled latexml csymbols (lines 440-681)
# but "conditional" is not in it

conditional_example = """
MathML for: k|n
<apply>
  <csymbol cd="latexml">conditional</csymbol>
  <ci>k</ci>
  <ci>n</ci>
</apply>
"""

# This should be added to the list of known latexml csymbols:
# Around line 500-600, in the condition checking if content in [list of symbols]:

fix_2_code = """
# Add "conditional" to the known latexml operators (around line 440-681)
# In the section: if content in ["annotated", "approaches-limit", ..., "weierstrass-p"]:
#   Add "conditional" to the list, then add:
#   elif content == "conditional":
#       retval = SemanticSymbol("O!divides", parent=parent)
#   
# Or more generally, add to the big list:
elif content == "conditional":
    retval = SemanticSymbol("O!divides", parent=parent)
"""

# Issue 3: Missing "differential-d" csymbol
# Represents the differential 'd' in integrals

differential_example = """
MathML for: ∫ f(x) dx
<csymbol cd="latexml">differential-d</csymbol>
"""

# Solution: Add to handled csymbols
fix_3_code = """
elif content == "differential-d":
    retval = SemanticSymbol("O!differential", parent=parent)
"""

# Issue 4: Missing "subscript" and "superscript" from "ambiguous" cd
# These appear in almost all formulas but don't cause failures
# The code does handle them in the ambiguous section (lines 700-701)

# The issue is that "subscript" and "superscript" create operators
# that should work fine. Let me verify by checking what the code does...

ambiguous_subscript = """
# Current code (lines 700-701):
elif cd == "ambiguous":
    if content == "subscript":
        retval = SemanticSymbol("O!SUB", parent=parent)
    elif content == "superscript":
        retval = SemanticSymbol("O!SUP", parent=parent)
"""

# These are already handled, so they're not the issue.
# The reason they appear in the PROBLEMATIC_CSYMBOLS list is just
# because they appear in the failing formulas, but they themselves
# don't cause the failure - it's the share/conditional/differential-d issues

print("Analysis complete. See fixes above.")
