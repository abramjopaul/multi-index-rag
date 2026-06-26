# Copyright (c) 2025 Abram Jopaul
# License: GNU GPLv3
#
# Formula-to-Embedding conversion using FastText training.
# Direct SLT/OPT representation trainer: skips latexmlmath subprocess entirely.
#
# Loads pre-computed MathML from TSV files and trains FastText on encoded tuples.
# Optimized for Google Compute Engine with chunked I/O and GCS support.

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

from multirag.config.path_configs import (
    FASTTEXT_MODEL_DIR,
    FORMULA_INDEX_DIR,
    OPT_REPRESENTATION,
    SLT_REPRESENTATION,
)
from multirag.embedding.formula_model_manager import FastTextModelManager
from multirag.formula_search import (
    MathExtractor,
    SymbolTree,
    TokenIDManager,
    TupleTokenizationMode,
    TupleTokenizer,
    encode_tuples,
    extract_tuples_from_mathml_direct,
)
from multirag.formula_search.encoder_maps import save_maps
from multirag.utils.file_utils import (
    USE_GCS_PATH_OPS,
    count_lines,
    delete_file,
    get_file_size,
    glob_files,
    makedirs,
    open_file,
    path_exists,
)

logger = logging.getLogger(__name__)


class TrainingProgressCallback(CallbackAny2Vec):
    """Callback to track FastText training progress with tqdm.
    
    Note: This class is needed for unpickling FastText models that were
    trained with a callback reference to this class. Required for gensim
    compatibility during model loading.
    """

    def __init__(self, epochs: int, corpus_file: str):
        """Initialize callback with epoch count and corpus file size."""
        self.epochs = epochs
        self.epoch = 0

        # Calculate corpus size (number of lines)
        with open_file(corpus_file, "r", encoding="utf-8") as f:
            self.corpus_size = sum(1 for _ in f)

        self.pbar = None

    def on_epoch_begin(self, model) -> None:
        """Called at start of each epoch."""
        self.epoch += 1
        total_words = (
            self.corpus_size * model.corpus_total_words
            if model.corpus_total_words
            else self.corpus_size
        )
        desc = f"Training (Epoch {self.epoch}/{self.epochs})"

        if self.pbar:
            self.pbar.close()

        self.pbar = tqdm(
            total=total_words, desc=desc, unit=" words", unit_scale=True, leave=True
        )

    def on_epoch_end(self, model) -> None:
        """Called at end of each epoch."""
        if self.pbar:
            self.pbar.update(self.pbar.total - self.pbar.n)  # Complete the bar
            self.pbar.close()

    def on_train_end(self, model) -> None:
        """Called when training finishes."""
        if self.pbar:
            self.pbar.close()

    def __getstate__(self) -> Dict:
        """Exclude pbar from pickling (tqdm with file handles can't be pickled)."""
        state = self.__dict__.copy()
        state["pbar"] = None
        return state

    def __setstate__(self, state: Dict) -> None:
        """Restore state, ensuring pbar is properly initialized."""
        self.__dict__.update(state)


class FormulaTrainerDirect:
    """
    Train FastText model on formula representations using pre-computed SLT/OPT MathML.

    Unlike FormulaTrainer, this class:
    1. Reads MathML directly from TSV files (no latexmlmath subprocess)
    2. Extracts trees directly from MathML via MathExtractor
    3. Is optimized for Google Compute Engine with chunked I/O

    Workflow:
    1. Load MathML from SLT/OPT TSV files (chunked reading for 50MB+ files)
    2. Convert MathML to SymbolTree (isolate_pmml/cmml → convert_to_layoutsymbol/semanticsymbol)
    3. Extract tuples using SymbolTree.get_pairs()
    4. Encode tuples with TokenIDManager (saves encoder maps)
    5. Save encoded corpus in LineSentence format (whitespace-separated tokens)
    6. Train FastText model with gensim
    7. Store training metadata

    EXAMPLE FLOW with LaTeX formula "2q^2":
    ==========================================
    
    INPUT (LaTeX):
      2q^2
    
    STEP 1: MathML Representation (from TSV)
    ----------------------------------------
      <math alttext="2q^{2}" class="ltx_Math" display="block" xmlns="http://www.w3.org/1998/Math/MathML">
        <semantics>
          <mrow>
            <mn>2</mn>
            <mo>⁢</mo>
            <msup>
              <mi>q</mi>
              <mn>2</mn>
            </msup>
          </mrow>
        </semantics>
      </math>
    
    STEP 2: Parse MathML → SymbolTree
    ---------------------------------
      • Extract presentation MathML (isolate_pmml)
      • Convert to LayoutSymbol tree structure
      • Represents spatial layout: 2 (number), q (variable), exponent relationship
    
    STEP 3: Generate Tuples from SymbolTree (window=2, eob=True)
    -----------------------------------------------------------
      SymbolTree.get_pairs() extracts structural relationships:
      
      Tuple Format: (node1, node2, relation1, relation2)
      - node1, node2: Tree node labels (N!number, V!variable, M!operator, 0!=EOB)
      - relation1: Left/parent relation (n=north, s=south, e=east, w=west, a=above, etc.)
      - relation2: Right/child relation (same as relation1, "-" for end-of-branch)
      
      Generated Tuples (5 total):
        1. (N!2, V!q, n, -)           ← Number 2 above variable q
        2. (N!2, N!2, na, -)          ← Number 2 with above-after relation
        3. (V!q, N!2, a, n)           ← Variable q with exponent 2 (above-north)
        4. (N!2, 0!, n, na)           ← Number 2 to end marker
        5. (V!q, 0!, n, n)            ← Variable q to end marker
    
    STEP 4: Encode Tuples → Unicode Characters
    ------------------------------------------
      Three-stage encoding process using TokenIDManager and encoder_maps:
      
      Stage A: Parse and Tokenize Tuple
      ─────────────────────────────────
        Raw tuple: "N!2\tV!q\tn\t-"
        Split by tab: ["N!2", "V!q", "n", "-"]
        Tokenize nodes (Both_Separated mode): ["N!", "2"] + ["V!", "q"] + edge "n" + edge "-"
        
      Stage B: Map Tokens → Numeric IDs (via encoder_maps)
      ───────────────────────────────────────────────────
        TokenIDManager.get_or_assign_id() looks up each token:
        
        Node tokens (ID space 60000+):
          "N!"   → 60000  (number type)
          "2"    → 60001  (specific number value)
          "V!"   → 60002  (variable type)
          "q"    → 60003  (specific variable q)
        
        Edge tokens (ID space 500+):
          "n"    → 500    (north relation)
          "-"    → 501    (end-of-branch)
        
        Result: Numeric sequence = [60000, 60001, 60002, 60003, 500, 501]
        
      Stage C: Convert Numeric IDs → Unicode Characters
      ──────────────────────────────────────────────────
        STEP C.1: Individual ID to character conversion
          60000 → chr(60000)  (U+EA60)
          60001 → chr(60001)  (U+EA61)
          60002 → chr(60002)  (U+EA62)
          60003 → chr(60003)  (U+EA63)
          500   → chr(500)    (U+01F4)
          501   → chr(501)    (U+01F5)
        
        STEP C.2: Concatenate all characters into single string
          [chr(60000), chr(60001), chr(60002), chr(60003), chr(500), chr(501)]
          → chr(60000) + chr(60001) + chr(60002) + chr(60003) + chr(500) + chr(501)
          → "ꙠꙡꙢꙣǴǵ"  (single encoded tuple string, no spaces)
        
        ✅ Result: ONE TUPLE → ONE ENCODED STRING "ꙠꙡꙢꙣǴǵ"
      
      EXAMPLES (from formula 2q^2):
      ──────────────────────────────
        Tuple 1: "N!2\tV!q\tn\t-"
          Tokens: ["N!", "2", "V!", "q", "n", "-"]
          Numeric: [60000, 60001, 60002, 60003, 500, 501]
          Unicode: "ꙠꙡꙢꙣǴǵ"  (1 encoded tuple)
        
        Tuple 2: "V!q\tN!2\ta\tn"
          Tokens: ["V!", "q", "N!", "2", "a", "n"]
          Numeric: [60002, 60003, 60000, 60001, 502, 500]
          Unicode: "ꙢꙣꙠꙡǶǴ"  (1 encoded tuple)
        
        Tuple 3: "N!2\t0!\tn\tna"
          Tokens: ["N!", "2", "0!", "n", "n", "a"]
          Numeric: [60000, 60001, 60004, 500, 500, 502]
          Unicode: "ꙠꙡꙤǴǴǶ"  (1 encoded tuple)
      
      Encoder maps saved to: encoder_maps_{tree_type}.tsv
        Maps node/edge IDs to components (N!2, V!q, n, -, etc.)
    
    STEP 5: Join Multiple Tuples with Whitespace
    ─────────────────────────────────────────────
      STEP 5.1: Collect all encoded tuples from formula
        encoded_tokens = ["ꙠꙡꙢꙣǴǵ", "ꙢꙣꙠꙡǶǴ", "ꙠꙡꙤǴǴǶ", ...]
      
      STEP 5.2: Join with whitespace separator
        encoded_sequence = " ".join(encoded_tokens)
        → "ꙠꙡꙢꙣǴǵ ꙢꙣꙠꙡǶǴ ꙠꙡꙤǴǴǶ ..."
        
        ✅ Result: ONE FORMULA → ONE LINE with SPACE-SEPARATED ENCODED TUPLES
    
    STEP 6: Final Corpus Format
    ────────────────────────────
      ✅ CORPUS FILE (one line per formula):
      
        Formula 1 (2q^2):     ꙠꙡꙢꙣǴǵ ꙢꙣꙠꙡǶǴ ꙠꙡꙤǴǴǶ
        Formula 2 (x + y):    ꙥꙦꙧꙨǴǷ ꙩꙪꙫꙬǶǸ ...
        Formula 3 (a^2+b^2):  ꙭꙮ꙯꙰Ǵǹ ꙱꙲꙳ꙴǺǻ ...
      
      Each line is treated by FastText as a "sentence" where:
      • Whitespace is the token boundary
      • Each space-separated unicode string is a "word" in FastText vocabulary
      • FastText learns embeddings for word pairs based on context window
      • Example: "ꙠꙡꙢꙣǴǵ" and "ꙢꙣꙠꙡǶǴ" co-occur within window → similar embeddings
    
    KEY INSIGHT:
    The encoding hierarchy:
    1. Individual node/edge elements → Numeric IDs (TokenIDManager)
    2. Numeric IDs → Unicode characters (chr())
    3. Concatenated characters → One encoded tuple string (no spaces)
    4. Multiple encoded tuples → One line in corpus (space-separated)
    5. FastText trains on corpus lines as "sentences" with tuples as "words"
    
    The tuple encoding preserves the mathematical structure:
    • Node labels (N!, V!, M!) capture semantic content (numbers vs. variables vs. operators)
    • Node values (2, q, +) capture specific symbols
    • Edge labels (n, s, e, w, a) capture spatial layout (above, beside, exponent, etc.)
    • Rare Unicode characters enable efficient FastText tokenization
    """

    def __init__(
        self,
        tree_type: Literal["SLT", "OPT", "SLT-TYPE"] = "SLT",
        embedding_type: Optional[TupleTokenizationMode] = None,
        vector_size: int = 300,
        window: int = 5,
        min_n: int = 10,
        max_n: int = 10,
        negative: int = 20,
        sg: int = 1,
        hs: int = 0,
        word_ngrams: int = 1,
        output_dir: Optional[str] = None,
        num_workers: int = 8,
        tokenize_number: Optional[bool] = None,
        chunk_size: int = 5000,
        epochs: int = 30,
    ):
        """
        Initialize FormulaTrainerDirect.

        Args:
            tree_type: "SLT" (Symbol Layout Tree), "OPT" (Operator Tree), or "SLT-TYPE"
            embedding_type: Node tokenization mode
                - None (auto): Both_Separated for SLT/OPT, Type for SLT-TYPE
                - Type: Extract only node types (for SLT-TYPE or custom)
                - Both_Separated: Extract type and value separately (for SLT/OPT)
            vector_size: FastText vector dimension (default: 300)
            window: Context window size (default: 5)
            min_n: Minimum n-gram size (default: 3)
            max_n: Maximum n-gram size (default: 6)
            negative: Number of negative samples (default: 20, 0 = use hierarchical softmax)
            sg: Training algorithm: 1 = Skip-gram, 0 = CBOW (default: 1)
            hs: Use hierarchical softmax: 1 = yes, 0 = no (default: 0, use negative sampling)
            word_ngrams: Number of word n-grams (default: 1)
            output_dir: Output directory (default: data/formula-indexing)
            num_workers: Number of worker threads (default: 4, None = auto-tune)
            tokenize_number: Whether to split numeric values
                - None (auto): True for SLT, False for OPT/SLT-TYPE
            chunk_size: Formulas per chunk when reading TSV (default: 5000, for GCE memory efficiency)
            epochs: Number of training epochs (default: 30)
        """
        self.tree_type = tree_type.upper()
        if self.tree_type not in {"SLT", "OPT", "SLT-TYPE"}:
            raise ValueError(
                f"tree_type must be 'SLT', 'OPT', or 'SLT-TYPE', got {self.tree_type}"
            )

        # Auto-configure embedding_type based on tree_type if not provided
        if embedding_type is None:
            self.embedding_type = (
                TupleTokenizationMode.Type
                if self.tree_type == "SLT-TYPE"
                else TupleTokenizationMode.Both_Separated
            )
        else:
            self.embedding_type = embedding_type

        self.vector_size = vector_size
        self.window = window
        self.min_n = min_n
        self.max_n = max_n
        self.negative = negative
        self.sg = sg
        self.hs = hs
        self.word_ngrams = word_ngrams
        self.epochs = epochs
        self.chunk_size = chunk_size

        # Set tokenize_number based on tree_type if not explicitly provided
        if tokenize_number is None:
            self.tokenize_number = self.tree_type == "SLT"
        else:
            self.tokenize_number = tokenize_number

        logger.info(
            f"Initialized DirectTrainer with tree_type={self.tree_type}, "
            f"embedding_type={self.embedding_type.name}, tokenize_number={self.tokenize_number}"
        )

        self.num_workers = num_workers

        # Setup output directories
        self.output_dir = Path(output_dir) if output_dir else FORMULA_INDEX_DIR
        makedirs(str(self.output_dir), exist_ok=True)

        # Artifact paths (use lowercase tree_type with hyphens replaced by underscores)
        tree_type_suffix = self.tree_type.lower().replace("-", "_")

        # Create tree-type-specific directory
        self.tree_type_dir = self.output_dir / tree_type_suffix
        makedirs(str(self.tree_type_dir), exist_ok=True)

        self.encoder_maps_path = (
            self.tree_type_dir / f"encoder_maps_{tree_type_suffix}.tsv"
        )
        self.corpus_path = self.tree_type_dir / f"corpus_{tree_type_suffix}.txt"
        self.model_path = self.tree_type_dir / f"fasttext_model_{tree_type_suffix}.bin"
        self.metadata_path = (
            self.tree_type_dir / f"training_metadata_{tree_type_suffix}.json"
        )
        self.checkpoint_path = (
            self.tree_type_dir / f"checkpoint_{tree_type_suffix}.json"
        )

        # Initialize tokenizer
        self.token_id_manager = TokenIDManager()
        self.tuple_tokenizer = TupleTokenizer(
            token_id_manager=self.token_id_manager,
            embedding_type=self.embedding_type,
            tokenize_number=self.tokenize_number,
        )

        # Select representation directory based on tree_type
        if self.tree_type in ("SLT", "SLT-TYPE"):
            self.representation_dir = SLT_REPRESENTATION
        else:  # OPT
            self.representation_dir = OPT_REPRESENTATION

        # Initialize model manager (lazy initialization for backward compatibility)
        self.model_manager = FastTextModelManager(
            model_path=str(self.model_path),
            metadata_path=str(self.metadata_path),
            corpus_path=str(self.corpus_path),
            vector_size=vector_size,
            window=window,
            min_n=min_n,
            max_n=max_n,
            negative=negative,
            sg=sg,
            hs=hs,
            word_ngrams=word_ngrams,
            num_workers=num_workers,
        )

        self.used_files: List[str] = []
        self.num_formulas_loaded = 0
        self.total_errors = 0  # Track errors during formula processing

    def load_formulas_from_tsv_chunks(
        self,
        file_numbers: Optional[List[int]] = None,
        num_formulas: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load MathML formulas from TSV files in chunks (memory-efficient for 50MB+ files).

        Args:
            file_numbers: List of file numbers to load (e.g., [1, 2, 3]).
                         If None, loads all files.
            num_formulas: Maximum number of formulas to load across all files.
                         If None, loads everything.

        Returns:
            DataFrame with columns: id, formula (MathML content)

        Example:
            >>> trainer = FormulaTrainerDirect(tree_type="SLT")
            >>> df = trainer.load_formulas_from_tsv_chunks(file_numbers=[1, 2], num_formulas=10000)
        """
        if not path_exists(str(self.representation_dir)):
            raise FileNotFoundError(
                f"Representation directory not found: {self.representation_dir}"
            )

        logger.info(f"Loading MathML formulas from {self.representation_dir}")

        # Determine which files to load
        if file_numbers is None:
            # Load all .tsv files, sorted by number
            all_files = glob_files(str(self.representation_dir), "*.tsv")
            file_numbers = sorted(
                [int(Path(f).stem) for f in all_files if Path(f).stem.isdigit()]
            )

        if not file_numbers:
            raise FileNotFoundError(f"No TSV files found in {self.representation_dir}")

        # Build file paths
        files = [self.representation_dir / f"{num}.tsv" for num in file_numbers]
        self.used_files = [f.name for f in files]

        logger.info(f"Loading {len(files)} files: {', '.join(self.used_files)}")

        all_formulas = []
        total_loaded = 0

        with tqdm(
            total=num_formulas or float("inf"),
            desc="Loading MathML",
            unit="formula",
        ) as pbar:
            for file_path in files:
                if not path_exists(str(file_path)):
                    logger.warning(f"File not found, skipping: {file_path}")
                    continue

                try:
                    # Read TSV in chunks for memory efficiency
                    for chunk in pd.read_csv(
                        str(file_path),
                        sep="\t",
                        usecols=["id", "formula"],
                        dtype={"id": int, "formula": str},
                        chunksize=self.chunk_size,
                        encoding="utf-8",
                    ):
                        if num_formulas and total_loaded >= num_formulas:
                            break

                        # Limit chunk if num_formulas is set
                        if num_formulas:
                            remaining = num_formulas - total_loaded
                            chunk = chunk.iloc[:remaining]

                        all_formulas.append(chunk)
                        total_loaded += len(chunk)
                        pbar.update(len(chunk))

                        if num_formulas and total_loaded >= num_formulas:
                            break

                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
                    continue

                if num_formulas and total_loaded >= num_formulas:
                    break

        if not all_formulas:
            raise ValueError("No formulas loaded from TSV files")

        combined_df = pd.concat(all_formulas, ignore_index=True)
        self.num_formulas_loaded = len(combined_df)

        logger.info(f"Total formulas loaded: {self.num_formulas_loaded}")
        logger.info(f"Files used: {', '.join(self.used_files)}")

        return combined_df

    def process_formulas_batch(
        self,
        formulas_df: pd.DataFrame,
        formula_column: str = "formula",
    ) -> List[str]:
        """
        Process batch of formulas: parse MathML and encode tuples.

        Args:
            formulas_df: DataFrame with formulas (id and formula columns)
            formula_column: Column name containing MathML strings

        Returns:
            List of encoded sequences (one per formula)
        """
        logger.info(
            f"Processing {len(formulas_df)} formulas ({self.tree_type} trees)..."
        )

        return self.process_latex_formula(formulas_df, formula_column)

    def process_latex_formula(
        self,
        formulas_df: pd.DataFrame,
        formula_column: str = "formula",
    ) -> List[str]:
        """Process formulas: parse MathML and encode tuples."""
        encoded_sequences = []
        successful = 0

        with tqdm(total=len(formulas_df), desc="Processing", unit="formula") as pbar:
            for _, row in formulas_df.iterrows():
                try:
                    mathml = row[formula_column]
                    if not mathml or not isinstance(mathml, str):
                        self.total_errors += 1
                        pbar.update(1)
                        continue

                    tuples = extract_tuples_from_mathml_direct(mathml, self.tree_type)  # type: ignore
                    if tuples:
                        encoded_seq = encode_tuples(tuples, self.tuple_tokenizer)
                        encoded_sequences.append(encoded_seq)
                        successful += 1
                    else:
                        self.total_errors += 1

                except Exception as e:
                    logger.debug(f"Error processing formula: {e}")
                    self.total_errors += 1

                pbar.update(1)

        logger.info(f"Processed: {successful} successful, {self.total_errors} failed")
        return encoded_sequences

    def save_corpus_and_maps(self, encoded_sequences: List[str]) -> None:
        """
        Save encoded sequences and encoder maps.

        Args:
            encoded_sequences: List of encoded sequences (whitespace-separated tokens)
        """
        logger.info("Saving corpus and encoder maps...")

        # Save corpus in LineSentence format
        makedirs(str(self.corpus_path.parent), exist_ok=True)
        with open_file(str(self.corpus_path), "w", encoding="utf-8") as f:
            for seq in encoded_sequences:
                f.write(seq + "\n")

        logger.info(f"Corpus saved: {self.corpus_path}")

        # Save encoder maps
        node_map = self.token_id_manager.node_map
        edge_map = self.token_id_manager.edge_map
        save_maps(node_map, edge_map, str(self.encoder_maps_path))
        logger.info(f"Encoder maps saved: {self.encoder_maps_path}")

    def train(
        self,
        file_numbers: Optional[List[int]] = None,
        num_formulas: Optional[int] = None,
        formula_column: str = "formula",
        epochs: Optional[int] = None,
    ) -> FastText:
        """
        Train FastText model on formulas from TSV representations.

        Args:
            file_numbers: List of file numbers to train on
            num_formulas: Maximum formulas to train on
            formula_column: Column name with MathML
            epochs: Number of training epochs (default: self.epochs from __init__)

        Returns:
            Trained FastText model

        Example:
            >>> trainer = FormulaTrainerDirect(tree_type="SLT", epochs=30)
            >>> model = trainer.train(file_numbers=[1, 2], num_formulas=10000)
        """
        if epochs is None:
            epochs = self.epochs

        logger.info(f"Starting training ({self.tree_type} trees, {epochs} epochs)...")

        # Load formulas
        formulas_df = self.load_formulas_from_tsv_chunks(file_numbers, num_formulas)

        # Process formulas
        encoded_sequences = self.process_formulas_batch(formulas_df, formula_column)

        if not encoded_sequences:
            raise ValueError("No formulas were successfully processed")

        # Save corpus and maps
        self.save_corpus_and_maps(encoded_sequences)

        # Train FastText from corpus file (memory-efficient) via manager
        return self.model_manager.train_from_corpus_file(
            epochs=epochs,
            num_sequences=len(encoded_sequences),
            tree_type=self.tree_type,
            embedding_type_name=self.embedding_type.name,
            tokenize_number=self.tokenize_number,
            used_files=self.used_files,
            num_formulas_loaded=self.num_formulas_loaded,
            encoder_maps_path=str(self.encoder_maps_path),
        )

    def train_from_corpus(
        self,
        corpus_path: str,
        epochs: Optional[int] = None,
    ) -> FastText:
        """
        Train FastText model directly from a pre-existing corpus file.
        
        This function streams from disk instead of loading all data into memory.
        Useful when corpus file already exists and memory is limited.

        Args:
            corpus_path: Path to corpus file (LineSentence format: one sentence per line)
            epochs: Number of training epochs (default: self.epochs from __init__)

        Returns:
            Trained FastText model

        Example:
            >>> trainer = FormulaTrainerDirect(tree_type="SLT")
            >>> model = trainer.train_from_corpus("./data/formula-indexing/slt/corpus_slt.txt", epochs=5)
        """
        if epochs is None:
            epochs = self.epochs

        if not path_exists(corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        logger.info(f"Training from corpus file: {corpus_path}")
        logger.info(f"Epochs: {epochs}")

        # Update manager's corpus path and train
        self.model_manager.corpus_path = Path(corpus_path)
        return self.model_manager.train_from_corpus_file(
            epochs=epochs,
            num_sequences=None,
            tree_type=self.tree_type,
            embedding_type_name=self.embedding_type.name,
            tokenize_number=self.tokenize_number,
            used_files=self.used_files,
            num_formulas_loaded=self.num_formulas_loaded,
            encoder_maps_path=str(self.encoder_maps_path),
        )

    def load_model(self, model_path: Optional[str] = None) -> FastText:
        """
        Load a trained FastText model.
        
        Delegates to FastTextModelManager for model loading.

        Args:
            model_path: Path to model file (default: self.model_path)

        Returns:
            Loaded FastText model
        """
        if model_path is not None:
            self.model_manager.model_path = Path(model_path)
        return self.model_manager.load()

    def get_sentence_vector(self, encoded_sequence: str) -> list:
        """
        Get vector representation for an encoded sequence.
        
        Delegates to FastTextModelManager for vector generation.

        Args:
            encoded_sequence: Whitespace-separated encoded tokens

        Returns:
            Vector representation (list of floats)
        """
        return self.model_manager.get_sentence_vector(encoded_sequence)

    def get_stats(self) -> Dict:
        """
        Get training statistics.
        
        Delegates to FastTextModelManager for stats retrieval.

        Returns:
            Dictionary with training metadata
        """
        return self.model_manager.get_stats()
