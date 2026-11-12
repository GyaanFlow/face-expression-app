"""Builds the examples folder from your own data.
usage: python make_examples.py path/to/unzipped_folder 3
Copies the photos of the first few people into examples/person_XX/ and writes one features csv per person.
Only use photos you are allowed to publish: a public Space shows them to everyone."""
import shutil
import sys
from pathlib import Path

import pandas as pd

src = Path(sys.argv[1])
k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
df = pd.read_csv(src / "features.csv", dtype={"person_id": str})
out = Path(__file__).parent / "examples"
for pid in list(dict.fromkeys(df["person_id"]))[:k]:
    g = df[df["person_id"] == pid]
    d = out / f"person_{pid}"
    d.mkdir(parents=True, exist_ok=True)
    for ph in g["photo_id"]:
        shutil.copy(src / "photos" / f"{ph}.jpg", d / f"{ph}.jpg")
    g.to_csv(out / f"person_{pid}.csv", index=False)
    print("wrote", d)
