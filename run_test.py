import sys, os, gc, random, logging, time
sys.path.insert(0, 'generator')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger()

from pycocotools.coco import COCO
from generator import CopyMoveGenerator
from stage_config import get_stage_config
import pandas as pd

cfg = get_stage_config('stage1', max_side=0, k_dest=30, k_ann=2)
sd = 'output/stage1_clean'
gen = CopyMoveGenerator(**cfg.to_generator_kwargs(
    output_dir_tampered=os.path.join(sd,'tampered'),
    output_dir_masks=os.path.join(sd,'masks')))

coco = COCO(os.path.join(sd, 'annotations_subset.json'))
try:
    stuff_coco = COCO('/home/harshita/coco/annotations/stuff_train2017.json')
except Exception as e:
    print(f"Failed to load stuff_coco: {e}")
    stuff_coco = None

ids = coco.getImgIds()
random.seed(333); random.shuffle(ids)

ok = []
skip_counts = {}
for img_id in ids:
    if len(ok) >= 20: break
    info = coco.loadImgs(img_id)[0]
    path = os.path.join('/home/harshita/coco/train2017', info['file_name'])
    
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
    t0 = time.time()
    r = gen.generate({'id':img_id,'file_name':info['file_name']}, path, anns, coco,
        stuff_coco=stuff_coco,
        use_perspective_scale=True, use_supercategory_pool=True)
    dt = time.time()-t0
    if isinstance(r, dict):
        ok.append(r)
        print(f'[{len(ok)}/20] {info["file_name"]} tier={r.get("placement_tier")} cat={r.get("source_category_name")} ({dt:.1f}s)', flush=True)
    else:
        reason = str(r).split(':')[1].strip() if ':' in str(r) else str(r)
        skip_counts[reason] = skip_counts.get(reason, 0) + 1
        if len(skip_counts) <= 5 or skip_counts[reason] <= 2:
            print(f'  skip: {reason} ({dt:.1f}s)', flush=True)
    gc.collect()

print(f'\nDone: {len(ok)} images generated', flush=True)
print(f'Skip breakdown: {skip_counts}', flush=True)

if ok:
    pd.DataFrame(ok).to_csv(os.path.join(sd,'metadata.csv'), index=False)
    from visualize import show_samples
    show_samples(os.path.join(sd,'metadata.csv'), '/home/harshita/coco/train2017', sd, max_samples=20)
