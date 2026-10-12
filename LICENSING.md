# Licensing — correction notice

_First issued 2026-09-13; revised 2026-10-12 for release v1.1. Supersedes the
licence statement of release v1.0._

## Summary

COCO-CMFD v1.0 was released under CC BY 4.0. That was incorrect: each sample is
an adaptation of one MS-COCO photograph, and those photographs carry their own
Creative Commons licences, which a single CC BY 4.0 grant cannot override.

- 4,026 v1.0 samples (32.8%) derive from photographs under a **NoDerivs**
  licence, which does not permit adaptations. They are removed.
- The remaining 8,245 samples are released in **four separately licensed
  parts**, one per licence of the source photographs.

| v1.1 part (config) | source photograph licence (COCO id) | samples |
|---|---|---:|
| `by` | Attribution 2.0 (4); No known copyright restrictions (7) | 2,006 |
| `by-nc` | Attribution-NonCommercial 2.0 (2) | 1,697 |
| `by-sa` | Attribution-ShareAlike 2.0 (5) | 1,092 |
| `by-nc-sa` | Attribution-NonCommercial-ShareAlike 2.0 (1) | 3,450 |
| removed | BY-NC-ND 2.0 (3), BY-ND 2.0 (6) | 4,026 |

The 2,006 `by` samples are 1,987 from Attribution photographs and 19 from
photographs with no known copyright restrictions.

## Why four parts rather than one licence

An earlier draft of this notice (2026-09-13) proposed relicensing the whole
retained set as CC BY-NC-SA 4.0. That does not work. A ShareAlike licence
requires adaptations to be shared under the same licence, so the 1,092
samples from BY-SA photographs cannot carry a NonCommercial term, and the
3,450 samples from BY-NC-SA photographs cannot be released without one. No
single licence satisfies both, so the release is a collection of parts, each
licensed as its source photographs require:

| part | terms that apply to its samples |
|---|---|
| `by` | attribution |
| `by-nc` | attribution, non-commercial use only |
| `by-sa` | attribution, adaptations under the same licence |
| `by-nc-sa` | attribution, non-commercial only, adaptations under the same licence |

Every row carries its source photograph's licence and attribution fields
(`coco_image_id`, `coco_license_id`, `license_name`, `license_url`,
`flickr_url`, `coco_url`, `modified`), read from the COCO annotation file, so a
user combining parts can see exactly which terms apply to each sample.

## Why this happened

MS-COCO's *annotations* are CC BY 4.0. MS-COCO's *images* are not. They are
Flickr photographs, each retaining its own Creative Commons licence, recorded
per image in the `license` field of `instances_train2017.json` and enumerated in
that file's `licenses` table:

| id | licence |
|---|---|
| 1 | Attribution-NonCommercial-ShareAlike |
| 2 | Attribution-NonCommercial |
| 3 | Attribution-NonCommercial-NoDerivs |
| 4 | Attribution |
| 5 | Attribution-ShareAlike |
| 6 | Attribution-NoDerivs |
| 7 | No known copyright restrictions |
| 8 | United States Government Work |

COCO aggregated these photographs; it did not relicense them. Reading "COCO is
CC BY" and applying that to derived images is the error, and it is a common one
in COCO-derived datasets. The upstream terms, with sources, are recorded in
[docs/upstream_licenses.md](docs/upstream_licenses.md).

## What changed

- **Release v1.1** is v1.0 minus the NoDerivs samples, split into the four parts
  above. No image, mask or metadata value of a retained sample is modified;
  only the licence and attribution columns are added
  (`tools/release_licence_subset.py`).
- **Per-sample licence manifest** (`tools/licence_manifest.py`): one row per
  sample with its source photograph and licence.
- **Generation filters on licence at photograph selection**
  (`generator/filter_annotations.py`), so future corpora are clean by
  construction rather than by subtraction. Excluding ids 3 and 6 leaves 80,111
  of 118,287 COCO train2017 photographs (67.7%) eligible. The filter is a
  module constant rather than a command-line flag, so a later run cannot forget
  it, and every generated row records the source photograph's
  `coco_license_id`.

## What this does not change

Models trained on v1.0 are not invalidated; their training set included
material that should not have been redistributed under the stated licence.
Data used for training stays local; only the licence-compatible subset is
redistributed.

## Verification

`tools/verify_release.py` checks the built release independently of the code
that produced it: per-part row counts, every row's licence against the COCO
annotation file, zero NoDerivs samples, the released set equal to v1.0 minus
NoDerivs, every v1.0 column byte-identical for retained rows, all four
ground-truth representations present and decodable, pixel changes against the
original COCO photograph confined to the labelled regions, and WebDataset
counts equal to parquet counts.
