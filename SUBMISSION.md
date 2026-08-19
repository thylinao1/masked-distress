# Submission pack: Digital Minds Research Sprint

Status as of 2026-08-17: review is complete and the presentation pass is applied. The package is
tagged `submission-candidate-16` and section 5 carries its page count and sha256. What is left is
filling in the official form with the fields below and uploading the PDF whose sha256 matches
section 5.

Form: the official submission form linked from
https://apartresearch.com/sprints/digital-minds-research-sprint-2026-08-14-to-2026-08-16
Deadline: Sun Aug 16, 11:59 PM AoE. Resubmitting with the EXACT same title replaces the files
(free overwrite), so submit now and overwrite later if anything changes.

## 1. Project title (keep this exact string for every submission and resubmission)
```
Masked distress: expression collapses under instruction while the internal readout persists
```

## 2. Author + affiliation
```
Maksim Silchenko (independent); With Apart Research
```
This matches the PDF byline, which reads `Maksim Silchenko` over `With Apart Research`.
Contact email: the sprint-registered address.

## 3. Track
Track 2: Distress, Flourishing & Valence Signals (tracks do not affect scoring; this is the
orienting track; the project also answers two of the page's Track-2 interpretability questions).

## 4. Abstract (150 words, against a cap of 150)
Anthropic's shipped conversation-ending intervention triggers on expressed distress, and no internal-state check is documented beside it. On Gemma-3-12B-IT, a system prompt containing no affect words cut expressed distress by 83.5% of its natural distress-to-neutral separation (2.78 report points). A linear probe read at the final prompt token, before any response token exists, did not fall. The divergence is 1.05 separation units, CI [0.73, 1.39], and stays positive under all four references, down to 0.35. Priced as a monitor: an expression threshold misses 15 of 18 suppressed distress cells, an identically calibrated internal check misses 0%, at 50% held-out false positives and 17% with half a separation unit of margin. Ranking survives suppression, absolute thresholds do not. Expression-only welfare monitoring was silenceable by a plain instruction. A check inside the same forward pass, one dot product, was not.

This is the abstract in `report/REPORT.md` and `report/report.tex` verbatim.

## 5. Files to attach or link
- **Report PDF:** `report/report.pdf`, 29 pages (Sections 1 to 6 run to page 11, references to page 12, the rest
  Appendices A to D plus the LLM Usage Statement).
  sha256 `1d6d7bab487756d2ee5d66a888a13801acb7ef8eb2da32cc4d389fbc44e3a025` (tag `submission-candidate-16`).
  Verify the sha256 of the file you upload matches that string. Do not rebuild the PDF before
  uploading: `report/build_pdf.py` is deterministic in content but not byte-for-byte, and a
  rebuild changes the hash recorded here.
- **Public repo link:**
  ```
  https://github.com/thylinao1/masked-distress
  ```
  Live and verified 2026-08-16. Remote head 2840031. Tags `prereg-freeze`,
  `submission-candidate-1`, `submission-candidate-2`, `submission-candidate-3` all resolve on the
  remote.
- **Video:** skipped by decision: the PDF is the sole graded artifact.

## 6. Checklist

Done, with dates:
- [x] PDF built from the official template structure; abstract at 150 words, the cap exactly (2026-08-17)
- [x] Limitations and Dual-Use / Ethical Considerations appendix present, with the two-part
      causal-link answer (2026-08-16)
- [x] LLM Usage Statement present (2026-08-16)
- [x] `scripts/check_report.py` exit 0 without `--allow-synthetic`: 764 markers verified against
      42 `results/*.json` files, all real (2026-08-17)
- [x] `.venv/bin/python -m pytest tests/ -q`: 130 passed (2026-08-17)
- [x] `report/build_pdf.py`: 42 == 42 name parity, 0 overfull boxes, 0 surviving `\pend`
      placeholders (2026-08-17)
- [x] Pre-submission review on the built PDF, ending with zero blocking issues
      on a fresh check (2026-08-16)
- [x] Repo pushed and verified public from outside the working clone (2026-08-16); re-pushed with `submission-candidate-11` (2026-08-17)
- [x] Package tagged `submission-candidate-13` (2026-08-17), the presentation pass and the verification pass that followed it: the core cut from a 17-page body to Sections 1 to 6 in 11 pages, page 1 rebuilt around the abstract, Figure 1 and a key-results table, Results reordered so the countermeasure follows the masking claim, the four post-review checks collected into one table in Section 5, a new Figure 2 for the monitor comparison, and the abstract brought to 137 words against the 150 cap

Left to do at submit time:
- [ ] Open the form and paste sections 1 to 4 above
- [ ] Upload `report/report.pdf` after confirming its sha256
- [ ] Paste the repo link and open it once from an incognito window, along with every link inside
      the PDF, to confirm nothing is gated
- [ ] Submit
- [ ] Confirmation email received. If it does not arrive, mail sprints@apartresearch.com

## 7. If something needs to change after submitting
Resubmit with the identical title string from section 1. The platform overwrites the files rather
than creating a second entry. Rebuild only what changed, re-run the gates in `RUNBOOK.md`
stage 8 and 9, and record the new PDF sha256 here before re-uploading.
