"""Compatibility entrypoint for checkpoint-based evaluation."""
from run import main

if __name__ == "__main__":
    main(default_action="validate")
