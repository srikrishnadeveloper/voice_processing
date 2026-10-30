import json, os, zipfile

ROOT = r'C:\Users\srik2\Desktop\extra\voice_processing'
SRC = os.path.join(ROOT, 'batches')
DST = os.path.join(ROOT, 'zbatches')

for b in ['b002', 'b003', 'b004', 'b005', 'b006']:
    src_dir = os.path.join(SRC, b)
    dst_dir = os.path.join(DST, b)
    os.makedirs(dst_dir, exist_ok=True)

    slug = f'tamil-calls-{b}'
    with open(os.path.join(dst_dir, 'dataset-metadata.json'), 'w') as f:
        json.dump({"title": slug, "id": f"srikrishnaos/{slug}",
                   "licenses": [{"name": "other"}]}, f)

    zip_path = os.path.join(dst_dir, f'{b}.zip')
    if not os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as z:
            for fn in sorted(os.listdir(src_dir)):
                if fn == 'dataset-metadata.json':
                    continue
                z.write(os.path.join(src_dir, fn), fn)
    mb = os.path.getsize(zip_path) / 1e6
    print(f'{b}: {zip_path}  {mb:.0f} MB  {zipfile.ZipFile(zip_path).namelist()[:2]}... ({len(zipfile.ZipFile(zip_path).namelist())} files)')
print('done')
