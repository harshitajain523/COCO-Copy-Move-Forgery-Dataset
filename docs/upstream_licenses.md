# Upstream licences

Verified 2026-09-28.

| Upstream | What COCO-CMFD uses | Licence | Source |
|---|---|---|---|
| MS-COCO 2017 annotations | instance polygons, categories (source regions, hard negatives) | CC BY 4.0 | COCO Terms of Use, https://cocodataset.org/#termsofuse |
| MS-COCO 2017 images | the one photograph each sample is built from | each photograph's own Flickr licence, per image in `instances_train2017.json` → `images[].license`, names and URLs in `licenses[]` | same page |
| COCO-Stuff annotations | stuff labels for the support-surface and horizon plausibility checks (all released stages set `use_stuff_annotations=True`) | CC BY 4.0 | https://github.com/nightrome/cocostuff, "Licensing" section |

COCO Terms of Use, as quoted: "The annotations in this dataset along with this
website belong to the COCO Consortium and are licensed under a Creative Commons
Attribution 4.0 License. The COCO Consortium does not own the copyright of the
images. Use of the images must abide by the Flickr Terms of Use."

The cocodataset.org page renders its terms client-side, so the text could not
be fetched as static HTML; it was confirmed through the search-indexed copy of
the page and several third-party quotations that agree word for word. The
COCO-Stuff statement was read directly from the repository README.

Both annotation sources must be credited in the dataset card.
