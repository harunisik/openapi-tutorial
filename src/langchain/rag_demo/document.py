from pathlib import Path
from langchain_core.documents import Document

def read_document(path: str | Path) -> Document:
    """
    Read a text file and return it as a LangChain Document with metadata.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        content = f.read()

    return Document(
        page_content=content,
        metadata={
            "source": str(path),
            "file_name": path.name,
            "suffix": path.suffix,
        },
    )

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)