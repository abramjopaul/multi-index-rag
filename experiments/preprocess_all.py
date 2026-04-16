from multirag.preprocessing.formula_tokenizer import FormulaTokenizer

tokenkizer = FormulaTokenizer(
    representation_type="slt", tokenization_mode="both_separated"
)
print(tokenkizer.extract_opt_tuples(r"$\\frac{4}{x}+\\frac{10}{y}=1$", window=2))
