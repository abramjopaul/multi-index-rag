import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from lxml import etree  # type: ignore
from tqdm import tqdm


@dataclass
class PostFormula:
    formula_id: str
    latex: str


@dataclass
class Answer:
    """A single answer post from Posts.V1.3.xml (PostTypeId=2)"""

    id: str
    parent_id: str  # ID of the question this answers
    score: int
    body_text: str  # plain text, formulas removed
    formulas: list[PostFormula] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Question:
    """A single question post from Posts.V1.3.xml (PostTypeId=1)"""

    id: str
    title: str
    score: int
    body_text: str
    tags: list[str]
    accepted_answer_id: str | None
    formulas: list[PostFormula] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class PostParser:
    """
    Streams Posts.V1.3.xml using lxml iterparse.
    Never loads the full file into memory.

    Args:
        xml_path: Path to Posts.V1.3.xml
        limit: Maximum number of posts to parse. If None, parse all posts.

    Usage:
        parser = PostParser("data/raw/collection/Posts.V1.3.xml", limit=1000)

        # Stream answers (memory efficient)
        for answer in parser.iter_answers():
            print(answer.id, answer.formulas)

        # Save both to JSONL in one pass
        parser.to_jsonl(
            answers_path="data/processed/collection/answers.jsonl",
            questions_path="data/processed/collection/questions.jsonl"
        )
    """

    def __init__(self, xml_path: str | Path, limit: int | None = None):
        self.xml_path = Path(xml_path)
        self.limit = limit

    def _parse_body(self, raw_html: str) -> tuple[str, list[PostFormula]]:
        """
        Parse HTML body → (plain_text, list of PostFormula).
        Uses lxml parser inside BeautifulSoup for speed.
        """
        if not raw_html:
            return "", []

        soup = BeautifulSoup(raw_html, "lxml")

        # Generator expression here is appropriate —
        # formulas per post are few, we materialise immediately into list
        formulas = [
            PostFormula(
                formula_id=span.get("id", ""),  # type: ignore
                latex=span.get_text().strip(),
            )
            for span in soup.find_all("span", class_="math-container")
        ]

        # for span in soup.find_all("span", class_="math-container"):
        #     span.decompose()

        # plain_text = soup.get_text(separator=" ").strip()
        # return plain_text, formulas
        body_text = str(soup.body.decode_contents()) if soup.body else str(soup)
        return body_text.strip(), formulas

    def _iter_rows(self):
        """
        Core private generator — yields raw lxml elements one at a time.
        This is the only place the file is read.
        Respects self.limit to stop after parsing N rows.
        """
        with open(self.xml_path, "rb") as f:
            context = etree.iterparse(f, events=("start",), tag="row", recover=True)
            count = 0
            for _, elem in context:
                if self.limit is not None and count >= self.limit:
                    break
                yield elem
                count += 1
                elem.clear()  # release memory immediately
                while elem.getprevious() is not None:  # clear preceding siblings too
                    del elem.getparent()[0]

    def iter_answers(self):
        """
        Public generator — yields Answer objects.
        Use this when you only need answers (e.g. building the corpus index).
        """
        for elem in tqdm(self._iter_rows(), desc="Parsing answers", unit=" posts"):
            if elem.get("PostTypeId") != "2":
                continue

            body_text, formulas = self._parse_body(elem.get("Body", ""))

            yield Answer(
                id=elem.get("Id", ""),
                parent_id=elem.get("ParentId", ""),
                score=int(elem.get("Score", 0)),
                body_text=body_text,
                formulas=formulas,
            )

    def iter_questions(self):
        """
        Public generator — yields Question objects.
        """
        for elem in tqdm(self._iter_rows(), desc="Parsing questions", unit=" posts"):
            if elem.get("PostTypeId") != "1":
                continue

            body_text, formulas = self._parse_body(elem.get("Body", ""))
            tags_raw = elem.get("Tags", "").replace("<", "").replace(">", " ")

            yield Question(
                id=elem.get("Id", ""),
                title=elem.get("Title", "").strip(),
                score=int(elem.get("Score", 0)),
                body_text=body_text,
                tags=[t.strip() for t in tags_raw.split() if t.strip()],
                accepted_answer_id=elem.get("AcceptedAnswerId"),
                formulas=formulas,
            )

    def to_jsonl(self, answers_path: str | Path, questions_path: str | Path) -> None:
        """
        Single pass over the XML — writes answers and questions to
        separate JSONL files simultaneously. Most efficient approach
        since reading the 7GB file twice would double the time.
        """
        answers_path = Path(answers_path)
        questions_path = Path(questions_path)
        answers_path.parent.mkdir(parents=True, exist_ok=True)
        questions_path.parent.mkdir(parents=True, exist_ok=True)

        n_answers = n_questions = 0

        with (
            open(answers_path, "w", encoding="utf-8") as af,
            open(questions_path, "w", encoding="utf-8") as qf,
        ):
            for elem in tqdm(
                self._iter_rows(),
                desc="Parsing Posts.V1.3.xml",
                unit=" posts",
                mininterval=1.0,  # update bar at most once per second (reduces overhead)
            ):
                post_type = elem.get("PostTypeId")
                body_text, formulas = self._parse_body(elem.get("Body", ""))

                if post_type == "2":  # Answer
                    record = Answer(
                        id=elem.get("Id", ""),
                        parent_id=elem.get("ParentId", ""),
                        score=int(elem.get("Score", 0)),
                        body_text=body_text,
                        formulas=formulas,
                    )
                    af.write(json.dumps(record.to_dict()) + "\n")
                    n_answers += 1

                elif post_type == "1":  # Question
                    tags_raw = elem.get("Tags", "").replace("<", "").replace(">", " ")
                    record = Question(
                        id=elem.get("Id", ""),
                        title=elem.get("Title", "").strip(),
                        score=int(elem.get("Score", 0)),
                        body_text=body_text,
                        tags=[t.strip() for t in tags_raw.split() if t.strip()],
                        accepted_answer_id=elem.get("AcceptedAnswerId"),
                        formulas=formulas,
                    )
                    qf.write(json.dumps(record.to_dict()) + "\n")
                    n_questions += 1

        print(f"Answers  → {answers_path}   ({n_answers:,})")
        print(f"Questions→ {questions_path} ({n_questions:,})")
