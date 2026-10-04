import sys
import os
import re

fn_input = sys.argv[1]
basename, extension = os.path.splitext(fn_input)
fn_output = basename + "_stripped" + extension


skipping = False
skip_indent = None

def skip(line):
    return bool(re.match(r"^\s*(\w+\.)*__\w+__", line))


with open(fn_input, "r", encoding="utf-8") as f:
    with open(fn_output, "w", encoding="utf-8") as f_out:
        for line in f:
            indent = len(line) - len(line.lstrip())
            if not skipping and skip(line):
                skipping = True
                skip_indent = indent
            elif skipping and indent <= skip_indent and not skip(line) and line.lstrip()[:3] not in ("'->", "`->", "-->"):
                skipping = False

            if not skipping:
                f_out.write(line)
