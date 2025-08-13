"""
This script checks commit message to match <type>[scope]: <description>
"""

import re
import sys

green_color = "\033[1;32m"
red_color = "\033[1;31m"
color_off = "\033[0m"
blue_color = "\033[1;34m"
yellow_color = "\033[1;33m"

commit_file = sys.argv[1]

regex = r"^[a-z]{1,}\[[a-z]{1,}\]: "

with open(commit_file, "r+") as file:
    commit_msg = file.read()

    if re.search(regex, commit_msg) or re.search(r'Merge|merge', commit_msg):
        print(f"{green_color}Commit message checking passed!{color_off}")
    else:
        print(f"{red_color}Bad commit {blue_color}{commit_msg}{color_off}")
        print(
            yellow_color
            + "Commit message format must match <type>[scope]: <description> "
            + "(add --no-verify to bypass)"
            + color_off
        )
        sys.exit(1)
