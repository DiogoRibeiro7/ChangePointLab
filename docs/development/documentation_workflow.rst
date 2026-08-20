Documentation Workflow
======================

Sphinx is the canonical user documentation build for ChangePointLab. Source
files live under ``docs/`` and the generated HTML is written to
``docs/_build/html``.

Run the canonical local build with:

.. code-block:: bash

   poetry run sphinx-build -W --keep-going -b html docs docs/_build/html

The ``-W --keep-going`` flags make documentation warnings fail the build while
still reporting the remaining warnings in the same run.

Executable Examples
-------------------

README and tutorial examples that must stay executable are marked with:

.. code-block:: html

   <!-- docs-example: execute -->

The marked Python block immediately following that marker is executed from a
temporary directory against the built wheel, not from the repository checkout.
Run the full local packaging and example check with:

.. code-block:: bash

   poetry build --clean
   poetry run python scripts/validate_docs_examples.py --dist-dir dist

Path Checks
-----------

Local Markdown links are checked with:

.. code-block:: bash

   poetry run python scripts/validate_docs_links.py

External links are intentionally excluded from the normal CI path because they
depend on network availability and third-party uptime.

Secondary API Inspection
------------------------

The pdoc build is retained only as a secondary API inspection artifact in CI.
It is generated under ``docs/_build/pdoc`` and should not be treated as the
published user documentation source of truth.
