def to_markdown_table(x: list[list[str]]) -> str:
    nl = "\n"

    content = nl
    content += f"| {' | '.join(x[0])} |"

    content += nl
    content += f"| {' | '.join(['---']*len(x[0]))} |"

    content += nl
    for entry in x[1:]:
        content += f"| {' | '.join(entry)} |{nl}"

    return content
