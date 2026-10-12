# TODO.md — COCO-CMFD

_Last updated: 2026-10-12 (release v1.1 built and verified)_

## Done
- Released dataset: 12,271 samples (strict 4,420 / relaxed_v1 166 /
  relaxed_v2 7,685) on HF `harshitajainn/coco-cmfd` and GitHub.
- **Forensic audit** → `DATASET_FORENSIC_AUDIT.md` (read-only; nothing
  generated, modified or trained). Scripts and per-sample outputs in
  `output/audit_forensic_2026-09-13/` (gitignored).

- **Release v1.1** (licence-corrected): 8,245 samples in four parts
  (`by`, `by-nc`, `by-sa`, `by-nc-sa`) built in `output/release_v1.1/`;
  `tools/verify_release.py` passes every check. README, dataset card,
  LICENSE-DATA, LICENSING.md and .zenodo.json updated.

## In progress
- Upload v1.1 to HF `harshitajainn/coco-cmfd`: replace `data/` and `wds/`
  with the per-part layout, push `docs/dataset_card.md` as README.md, and
  remove the v1.0 shards. Awaiting go-ahead.

## Needs user decision
1. **Training split leakage** (CRITICAL): the training repo's split keys on
   `sample_id`, so 57% of curated-test photographs also appear in train.
   Re-split grouped by `image_id` (audit X0) before any further data
   decision is judged on curated val/test.
3. **First extension to test** (audit section I): proposed X2 zero-training
   resampling probe on `r768`, then X1 area tails. Not started.
4. **Release docs corrections**: README / dataset card state bbox area as
   "pasted region size", call hard negatives "visually similar", and give
   incomplete regeneration steps (audit C12–C14).
5. **Unfinished shard**: recovery_v2 shard 5/6 never completed — leave as is
   or document it in the release.

## Blocked
- CoMoFoD ground-truth masks and DEFACTO are not on this machine; their
  per-sample statistics and COCO photo overlap with DEFACTO are unverified.

## Backlog
- Record per-sample mask provenance (SAM / GrabCut / polygon) and seed
  stream in any future generation run.
- Fix manifest provenance (commit the code before generating; one manifest
  per shard/pass rather than last-writer-wins).
