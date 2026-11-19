"""Reproducible, aggregate-only demographic preview; original sources stay intact."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

BANDS = ['＜18岁', '18–39岁', '40–59岁', '60–79岁', '≥80岁']
ALIASES = [
    ['脂肪组织'], ['毛细血管瘤'], ['皮脂腺腺瘤', '皮脂腺瘤'],
    ['皮脂腺癌', '皮脂腺腺癌'], ['睑板腺癌'], ['纤维瘤'], ['翼状胬肉'],
    ['肉芽肿'], ['脂溢'], ['痣'], ['鳞状细胞癌'], ['黄色肉芽肿'],
    ['基底细胞癌'], ['慢性炎症', '慢性炎性'], ['疣'], ['血管瘤'],
    ['黄色瘤'], ['囊肿', '囊瘤'], ['黑色素瘤'], ['汗腺囊'], ['乳头状'], ['钙化上皮瘤'],
]

def normalize_id(value):
    if isinstance(value, (float, int)) and math.isfinite(value) and value == int(value):
        return str(int(value))
    return str(value).strip()

def demographics(row):
    """Age/sex are transposed part-way through the source file."""
    values = row[1:3]
    sex_values = {str(v).strip() for v in values} & {'男', '女'}
    sex = next(iter(sex_values)) if len(sex_values) == 1 else None
    ages = [float(v) for v in values if isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 120]
    return sex, ages[0] if len(ages) == 1 else None

def age_band(age):
    if age is None:
        return None
    return sum(age >= boundary for boundary in [18, 40, 60, 80])

def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

def summarize(records, total):
    sex = Counter(r['sex'] for r in records)
    ages = [r['age'] for r in records if r['age'] is not None]
    bins = [sum(age_band(age) == i for age in ages) for i in range(5)]
    peak = max(bins, default=0)
    return dict(total=total, matched=len(records), male=sex['男'], female=sex['女'],
                sex_missing=total-sex['男']-sex['女'], age_known=len(ages),
                age_missing=total-len(ages), bands=bins, q1=quantile(ages,.25),
                median=quantile(ages,.5), q3=quantile(ages,.75), peak_n=peak,
                peak_label='、'.join(BANDS[i] for i,n in enumerate(bins) if n == peak) if peak else '无有效年龄')

def load_source(path):
    import xlrd
    sheet = xlrd.open_workbook(path).sheet_by_index(0)
    lookup = {}
    blank = 0
    swapped = 0
    for index in range(1, sheet.nrows):
        row = sheet.row_values(index)
        key = normalize_id(row[0])
        if not key:
            blank += 1
            continue
        if key in lookup:
            raise ValueError('Duplicate source identifier; resolve before joining')
        sex, age = demographics(row)
        swapped += str(row[1]).strip() in {'男','女'}
        lookup[key] = dict(sex=sex, age=age, diagnosis=str(row[5]).strip(), source_row=index+1)
    return lookup, dict(source_records=len(lookup), blank_id_rows=blank, swapped_rows=swapped)

def analyze(source, manifest, classes, controls):
    lookup, metadata = load_source(source)
    with open(manifest, encoding='utf-8-sig', newline='') as f:
        images = [r for r in csv.DictReader(f) if r['include_in_experiments']=='True']
    # Never merge distinct anonymous patients or coerce embedded codes into Excel IDs.
    patients = {(r['class_id'],r['patient_id']):r for r in images}
    class_list = json.loads(Path(classes).read_text(encoding='utf-8'))['classes']
    with open(controls, encoding='utf-8-sig', newline='') as f:
        expected = {r['class_id']:r for r in csv.DictReader(f)}
    result, flags, all_records = [], [], []
    for c in class_list:
        cid = c['class_id']
        cohort = [r for (code,_),r in patients.items() if code==cid]
        matches = []
        id_matches = 0
        for p in cohort:
            if p['patient_id_source'] != 'filename_prefix':
                continue
            record = lookup.get(p['patient_id_raw_token'])
            if record is None:
                continue
            id_matches += 1
            if not any(word in record['diagnosis'] for word in ALIASES[int(cid[1:])]):
                flags.append(dict(class_id=cid, source_row=record['source_row'], diagnosis=record['diagnosis'], reason='编号可匹配，病理文字未通过类别关键词核验，暂不纳入人口学统计'))
                continue
            matches.append(record)
        image_n = sum(r['class_id']==cid for r in images)
        assert len(cohort)==int(expected[cid]['patient_count'])
        assert image_n==int(expected[cid]['included_image_count'])
        result.append(dict(class_id=cid, name=c['class_name'], category=c['category'],
                           images=image_n, id_matches=id_matches, flagged=id_matches-len(matches),
                           **summarize(matches,len(cohort))))
        all_records.extend(matches)
    overall = dict(class_id='合计', name='全部病种', category='', images=len(images),
                   id_matches=sum(r['id_matches'] for r in result), flagged=len(flags),
                   **summarize(all_records,len(patients)))
    metadata['source_sha256'] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    metadata['manifest_sha256'] = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    return dict(metadata=metadata, bands=BANDS, classes=result, overall=overall, flags=flags)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--manifest', default='outputs/data_audit/patient_manifest.csv')
    parser.add_argument('--classes', default='configs/classes.json')
    parser.add_argument('--controls', default='outputs/data_audit/class_patient_summary.csv')
    parser.add_argument('--output', required=True)
    parser.add_argument('--xlrd-path')
    args=parser.parse_args()
    if args.xlrd_path:
        import sys
        sys.path.insert(0,args.xlrd_path)
    data=analyze(args.source,args.manifest,args.classes,args.controls)
    Path(args.output).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(metadata=data['metadata'], overall=data['overall'], flags=len(data['flags'])),ensure_ascii=False))
