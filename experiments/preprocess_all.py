from multirag.config.path_configs import (
    ANSWERS_JSONL,
    POSTS_XML,
    QREL_2022_JSONL,
    QREL_TASK1_2022_OFFICIAL,
    QUESTIONS_JSONL,
    TOPICS_JSONL,
    TOPICS_XML,
)
from multirag.preprocessing.post_parser import PostParser
from multirag.preprocessing.topic_parser import TopicReader

# from multirag.preprocessing.qrel_loader  import QrelReader

# --- Posts (one pass, ~10-15 min) ---
PostParser(POSTS_XML).to_jsonl(
    answers_path=ANSWERS_JSONL,
    questions_path=QUESTIONS_JSONL,
)
