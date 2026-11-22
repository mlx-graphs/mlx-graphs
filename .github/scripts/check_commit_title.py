"""Validate titles against the commit types used to calculate release versions."""

import re
import sys
from pathlib import Path

import tomllib

SKIP_CI = re.compile(r"\[(?:skip ci|ci skip|no ci|skip actions|actions skip)\]", re.I)


def valid_title(title, commit_types):
    types = "|".join(map(re.escape, commit_types))
    pattern = rf"(?:{types})(?:\([^()\s]+\))?!?: \S[^\r\n]*"
    return re.fullmatch(pattern, title) is not None and not SKIP_CI.search(title)


if __name__ == "__main__":
    with Path("pyproject.toml").open("rb") as project:
        config = tomllib.load(project)["tool"]["semantic_release"]
    commit_types = config["commit_parser_options"]["allowed_tags"]
    if not valid_title(sys.argv[1], commit_types):
        print("::error::Use a title like 'fix(loaders): handle empty graphs'.")
        print(f"Allowed types: {', '.join(commit_types)}. Omit CI skip markers.")
        sys.exit(1)
