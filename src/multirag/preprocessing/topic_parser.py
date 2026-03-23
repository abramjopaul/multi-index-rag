import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from lxml import etree  # type: ignore


@dataclass
class TopicFormula:
    formula_id: str
    latex: str


@dataclass
class Topic:
    """
    Represents a single ARQMath Task 1 query topic.
    Each topic is a mathematical question from Math StackExchange
    used as a query in the evaluation benchmark.
    """

    topic_id: str
    title: str
    question: str  # plain text, formulas removed
    tags: list[str]
    formulas: list[TopicFormula] = field(
        default_factory=list
    )  # LaTeX formulas in the question

    def to_dict(self) -> dict:
        return asdict(self)


class TopicReader:
    """
    Reads ARQMath Task 1 topic XML file into a map of Topic objects.
    Extracts plain text and LaTeX formulas separately from the question body.

    Usage:
        reader = TopicReader("data/raw/task1/Topics_2020.xml")
        topic  = reader.get_topic("A.1")
        print(topic.title)
        print(topic.formulas)

        # Save to JSONL
        reader.to_jsonl("data/processed/topics/topics.jsonl")
    """

    def __init__(self, topic_file_path: str | Path):
        self.topic_file_path = Path(topic_file_path)
        self.map_topics: dict[str, Topic] = self._read_topics()

    def _parse_formulas_text(self, raw_html: str) -> tuple[str, list[TopicFormula]]:
        """
        Takes raw HTML question body, returns:
        - plain text with formulas removed
        - list of TopicFormula extracted from math-container spans
        """
        if not raw_html:
            return "", []

        soup = BeautifulSoup(raw_html, "lxml")

        # Extract formulas before removing spans
        formulas = [
            TopicFormula(
                formula_id=span.get("id", ""),  # type: ignore
                latex=span.get_text().strip(),
            )
            for span in soup.find_all("span", class_="math-container")
        ]

        # TODO: We could keep formulas in the title and question for now
        # Remove formula spans so plain text is clean
        # for span in soup.find_all("span", class_="math-container"):
        #     span.decompose()

        plain_text = soup.get_text(separator=" ").strip()
        return plain_text, formulas

    def _read_topics(self) -> dict[str, Topic]:
        """
        Parse Topics XML. Topics are small (100 entries) so
        we load all into memory as a dict — no streaming needed.
        """
        map_topics: dict[str, Topic] = {}

        tree = etree.parse(
            str(self.topic_file_path)
        )  # lxml parse — fine for small file
        root = tree.getroot()

        for child in root:
            topic_id = child.attrib["number"]
            title_raw = child.findtext("Title", default="").strip()
            question_raw = child.findtext("Question", default="").strip()
            tags_raw = child.findtext("Tags", default="")

            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

            title_text, title_formulas = self._parse_formulas_text(title_raw)

            question_text, formulas = self._parse_formulas_text(question_raw)

            map_topics[topic_id] = Topic(
                topic_id=topic_id,
                title=title_text,
                question=question_text,
                tags=tags,
                formulas=title_formulas + formulas,
            )

        return map_topics

    def get_topic(self, topic_id: str) -> Topic:
        return self.map_topics[topic_id]

    def iter_topics(self):
        """Generator — yields Topic objects one at a time."""
        yield from self.map_topics.values()

    def to_jsonl(self, output_path: str | Path) -> None:
        """
        Write all topics to a JSONL file.
        Each line is one JSON object representing one Topic.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for topic in self.map_topics.values():
                f.write(json.dumps(topic.to_dict()) + "\n")

        print(f"Saved {len(self.map_topics)} topics → {output_path}")
