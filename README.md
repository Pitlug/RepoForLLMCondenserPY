# RepoForLLMCondenserPY
I had chatgpt 5 (copilot) create for me a python file that takes an entire repo/directory and collapses all code files and text files into readable text for the LLM to input for it's context window as one text file. (I have not done extensive testing on this tool so let me know what the limits are or any changes that should be made.)
# How to use
Run the py file with this syntax:
python3 repo_to_text.py </path/to/directory/repo> -o <repo-dump-output.txt>
## Modiifcations
Within the python file you can set what file types you want to read and what file types you don't want to read within the DEFAULT_IGNORED_DIRS, ALWAYS_SKIP_EXTS, PREFER_TEXT_EXTS, EXT_TO_LANG dictonaries.And there are a few other options in their.
