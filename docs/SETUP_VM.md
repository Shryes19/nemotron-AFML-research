# VM setup (CPU only)

Tested on Ubuntu 24.04, 6 cores, 7.7 GB RAM. Needs `git`, `wget`, `unzip`, Java 17 (`java -version`) and ~20 GB free disk (alfworld pulls in torch).

```bash
git clone https://github.com/PSaanviBhat/specialist-or-generalist-wm.git ~/wm
cd ~/wm
bash scripts/setup_vm.sh                         # ~1 h on a slow link; creates ~/w2w/uv_alfworld (Py 3.9) and ~/w2w/uv_sciworld (Py 3.8)
~/w2w/uv_alfworld/bin/python scripts/fetch_data.py   # ALFWorld games (Word2World grammar) + split files
```

If Java is missing: `sudo apt-get install -y openjdk-17-jre-headless`.

## Check it works

```bash
~/w2w/uv_alfworld/bin/python -m pytest tests/ --env alfworld -q -p no:warnings
~/w2w/uv_sciworld/bin/python -m pytest tests/ --env sciworld -q -p no:warnings
```

## Notes

- Word2World's ALFWorld grammar renames `put` to `move` and adds `help`; the stock `alfworld-download` data is not compatible.
- ALFWorld train ids are game names (`<task_type>_trial_...`); test ids are `alfworld_N`. ScienceWorld ids are `sciworld_N`. Both index AgentGym's own game lists.
- Each env runs in its own venv, so run each env's commands with that venv's python.
