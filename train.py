"""Compatibility entrypoint; all training options are defined by run.py."""
from run import main

if __name__ == "__main__":
    main(default_action="train")
