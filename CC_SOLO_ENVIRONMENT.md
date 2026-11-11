# Reproducible local environment

Source: https://github.com/petl-developers/petl, tag v1.7.17,
commit 077a74f3f0bf33abfa7559a16a73396da1c50c87 (MIT).
This is a public evaluation baseline derived from that project, using synthetic
local data. It does not represent a private production installation.

Use Python 3.11 (the evaluation container supplies Python 3.11.2):

```sh
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/petl-get-pip.py
python3 /tmp/petl-get-pip.py --user --break-system-packages pip==24.3.1
python3 -m pip install --user --break-system-packages -r requirements-cc-solo.txt
python3 -m pip install --user --break-system-packages --no-build-isolation --no-deps -e .
python3 -m pytest -q petl/test/transform/test_sorts.py petl/test/transform/test_joins.py
```

The editable installation generates the ignored `petl/version.py`; do not treat
its absence in a fresh Git checkout as a missing source module. No databases,
remote services or accounts are needed for local transform tests. Optional
format/database integrations have separate dependencies and are outside this
minimal environment. Report the tests relevant to the requested behavior and
their actual outcomes.

The fixed container has neither pip nor ensurepip preinstalled, so the commands
bootstrap pip from the official PyPA endpoint into the container user's home.
They do not alter the image or host Python. This setup requires network access
for dependency installation; subsequent local transformation tests do not.
