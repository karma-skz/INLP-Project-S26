Mid-submission report (ACL format)
=================================

Files:
- `mid_submission.tex` — ACL-format LaTeX file for the mid-term submission.
- `references.bib` — small BibTeX file with placeholder entries (replace with canonical BibTeX entries).

How to compile (from project root):
```bash
source inlp/bin/activate
cd reports
pdflatex mid_submission.tex
bibtex mid_submission
pdflatex mid_submission.tex
pdflatex mid_submission.tex
```

Notes:
- The document uses the ACL style files in `../acl-style-files/` (already present in the repo).
- Figures are referenced from `../figures/main_output/` — ensure those files exist.
- Replace placeholder BibTeX entries in `references.bib` with full citations before final submission.
