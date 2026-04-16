import time

import numpy
import numpy as np
from logging_config import configure_logging

from multirag.embedding.formula_embedder import FormulaEmbedder
from multirag.embedding.formula_trainer import FormulaTrainer
from multirag.formula_search.formula_tokenizer_pipeline import \
    FormulaTokenizerPipeline
from multirag.formula_search.opt_generator import OPTGenerator
from multirag.formula_search.slt_generator import SLTGenerator

# Configure logging (accepts CLI arg or defaults to INFO)
configure_logging()

# latex_formula = r"$$ \\{ r \\in \\mathbb Q \\mid r^2 >2, r>0 \\}$$"
# latex_formula = r"$A \\in \\mathbb{R}^{2 \\times 2}$"
latex_formula = r"x - y^2 = 0"

# optGen = OPTGenerator()
# tuples = optGen.generate_tuples(latex_formula=latex_formula, window=2, eob=True)
# print(f"{tuples} \n")

# sltGen = SLTGenerator()
# tuples = sltGen.generate_tuples(latex_formula=latex_formula, window=2, eob=True)
# print(f"{tuples} \n")

# pipeline = FormulaTokenizerPipeline()
# encoded = pipeline.tokenize_formula(latex_formula=latex_formula, tree_type="SLT", window=2, include_eob=True)
# print(f"{encoded} \n")
# pipeline.save_encoder_maps("./encoder_map1.tsv")

# test_encoded = '\uea60\uea61\uea60\uea63ǴǴ'
# token_ids = [ord(c) for c in test_encoded]
# print(f"Token IDs: {token_ids}")

print("\n" + "=" * 70)
print("STARTING FORMULA TRAINING WITH TIMING")
print("=" * 70 + "\n")

start_time = time.time()

trainer = FormulaTrainer(tree_type="OPT",use_process_pool=True, num_workers=8)
model = trainer.train(
    file_numbers=[1, 2, 3],
    num_formulas=50,
    formula_column="formula",
)

end_time = time.time()
elapsed_time = end_time - start_time

print("\n" + "=" * 70)
print(f"TRAINING TIME: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print("=" * 70 + "\n")

model = trainer.load_model(
    "/Users/abramjopaul/Documents/projects/multi-index-rag/data/formula-indexing/fasttext/fasttext_model_slt.bin"
)
print(np.shape(model.wv["\uea60\uea61\uea60\uea63ǴǴ"]))
print(model)
