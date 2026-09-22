# Third-party data notice

The MIT licence in `../LICENSE` covers the code and the POMA-produced artifacts in this
repository. The items below are third-party data with their own terms.

## Databricks OfficeQA (CC BY-SA 4.0)

The following are taken from, or derived from, the OfficeQA benchmark by Databricks
(https://github.com/databricks/officeqa, https://huggingface.co/datasets/databricks/officeqa),
licensed under Creative Commons Attribution-ShareAlike 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). Copyright Databricks, Inc.

- `databricks/*.txt` — Databricks' transformed text rendering of 14 Treasury Bulletin issues,
  redistributed unchanged. Used here as the input for the naive-chunking baseline.
- `officeqa.csv` — 20 questions, answers, UIDs, source references and difficulty labels,
  selected from the OfficeQA question set.
- `evidence_requirements.json` — per-question answers and source references from OfficeQA,
  plus POMA-authored evidence-index annotations.

Changes made: subset selection (20 of 246 questions, 14 of 697 documents); addition of
per-method evidence indices. No text of the Databricks files was modified.

These files are redistributed under the same CC BY-SA 4.0 terms, not under MIT. Databricks
distributes the full dataset gated so that answer keys stay out of web-crawled training and
agent search results; if you build on this repository, please do not spread the answers further.

## U.S. Treasury Bulletins (public domain)

The underlying Treasury Bulletin PDFs are works of the United States Government, obtained via
FRASER (https://fraser.stlouisfed.org/title/treasury-bulletin-407), and are in the public
domain. The POMA (`poma/`) and Unstructured.io (`unstructured/`) outputs were produced from
these PDFs and are covered by the repository MIT licence.
