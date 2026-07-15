"""CLI: speak text from arguments or stdin.

    python -m voice_agent.speak "Hello, I'm glad you're here."
    some_brain_process | python -m voice_agent.speak

When reading from a pipe, each blank-line-separated block is spoken as it
arrives, so a streaming brain can talk through this continuously.
"""

import sys

from .presenter import Presenter


def main():
    presenter = Presenter()

    if len(sys.argv) > 1:
        presenter.speak(" ".join(sys.argv[1:]))
        return

    if sys.stdin.isatty():
        print("Type what the agent should say (blank line to speak, Ctrl+C to quit):")

    block: list[str] = []
    try:
        for line in sys.stdin:
            if line.strip():
                block.append(line)
            elif block:
                presenter.speak("".join(block))
                block = []
        if block:
            presenter.speak("".join(block))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
