import subprocess
import re


def get_pylint_score(filepath):

    try:

        result = subprocess.run(
            ["pylint", filepath, "--score=y"],
            capture_output=True,
            text=True
        )

        output = result.stdout + result.stderr

        match = re.search(r"rated at ([0-9\.]+)/10", output)

        if match:
            return float(match.group(1))

        return 0.0

    except Exception:
        return 0.0