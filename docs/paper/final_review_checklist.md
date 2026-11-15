# Final paper review checklist — unresolved items only

- [ ] **Missing action-lookup baseline artifact.** The G-versus-action-lookup
  comparison remains unrun. Supply a scored action-lookup JSONL and a
  schema-compatible scored G JSONL with overlapping `id` values and the same
  numeric utility field, or retain the explicit limitation in the paper.
- [ ] **Missing raw G artifacts.** Recover the original `g_prompts/` and
  `g_candidates/` directories (including `pred_next`) from the GPU machine or
  experiment archive. The retained scored JSONLs cannot reconstruct generated
  text; do not claim raw-prediction reproducibility until these files are
  recovered.
- [ ] **Submission metadata and style.** Confirm title, author order,
  affiliations, course/venue, and the required citation/bibliography format.
  The current author-year prose bibliography is internally consistent and
  verified, but the repository has no venue-specific BibTeX/LaTeX file.
- [ ] **Venue conversion.** Replace Markdown image links with the target
  venue's figure environment and render-check the complete manuscript. If the
  venue requires a standalone detailed results table, copy the full finalized
  95% confidence intervals from `results_discussion.md` rather than shortening
  them.
- [ ] **Final team sign-off.** Obtain owner approval for the technical D
  description and ablation wording, finalized results/significance and
  limitations, GPU accounting/RQ4 interpretation, and whether Figure 3 remains
  despite the absence of a D-win feature-matrix row.
