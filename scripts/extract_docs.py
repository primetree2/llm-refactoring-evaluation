import ast
import tokenize
from io import StringIO


def extract_docstrings(filepath):
    """
    Extract docstrings from Python AST nodes.
    """

    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)

    docs = []

    module_doc = ast.get_docstring(tree)
    if module_doc:
        docs.append(module_doc)

    for node in ast.walk(tree):

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):

            doc = ast.get_docstring(node)

            if doc:
                docs.append(doc)

    return docs


def extract_comments(filepath):
    """
    Extract comments using Python tokenizer.
    """

    comments = []

    with open(filepath, "r", encoding="utf-8") as f:

        tokens = tokenize.generate_tokens(StringIO(f.read()).readline)

        for token in tokens:

            if token.type == tokenize.COMMENT:
                comments.append(token.string)

    return comments


def extract_documentation(filepath):
    """
    Return all documentation text (comments + docstrings).
    """

    docstrings = extract_docstrings(filepath)
    comments = extract_comments(filepath)

    combined = docstrings + comments

    return " ".join(combined)