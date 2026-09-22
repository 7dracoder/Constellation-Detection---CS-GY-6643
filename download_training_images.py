#!/usr/bin/env python3
"""Download a bounded, attributed external crop-training corpus from ESO."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from html.parser import HTMLParser

import cv2
import numpy as np

SOURCES = (
    ("eso0106a", "AURA", "The Milky Way star field around CS 31082-001"),
    ("eso1427a", "ESO", "The dark cloud Lupus 4"),
    ("eso1439a", "ESO/G. Beccari", "The colourful star cluster NGC 3532"),
    ("eso1242a", "ESO/VVV Survey/D. Minniti; Acknowledgement: Ignacio Toledo, Martin Kornmesser", "VISTA central Milky Way mosaic"),
)
LICENSE = "https://www.eso.org/public/copyright/"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        href = dict(attrs).get("href", "")
        if tag == "a" and href.endswith(".jpg"):
            self.links.append(href)


def get(url, limit):
    request = urllib.request.Request(url, headers={"User-Agent": "ConstellationCourseResearch/1.0"})
    with urllib.request.urlopen(request, timeout=90) as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Download exceeds byte limit: {url}")
    return data


def download(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    by_id = {record["id"]: record for record in old}
    records = []
    for identifier, credit, title in SOURCES:
        target = output / f"{identifier}.jpg"
        prior = by_id.get(identifier)
        if target.exists() and prior and hashlib.sha256(target.read_bytes()).hexdigest() == prior["sha256"]:
            prior["credit"] = credit
            records.append(prior)
            print("cached", identifier, flush=True)
            continue
        page = f"https://www.eso.org/public/images/{identifier}/"
        parser = Links()
        parser.feed(get(page, 2_000_000).decode())
        # Publication-sized photographs retain many resolved stars without
        # downloading hundreds of megabytes or a multi-gigapixel original.
        links = [url for url in parser.links if "/publicationjpg/" in url]
        if not links:
            links = [url for url in parser.links if "/large/" in url]
        if not links:
            raise ValueError(f"No public JPEG download on {page}")
        url = links[0]
        if url.startswith("/"):
            url = "https://www.eso.org" + url
        blob = get(url, 30_000_000)
        image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or min(image.shape) < 128:
            raise ValueError(f"Unusable image: {url}")
        target.write_bytes(blob)
        records.append(dict(id=identifier, title=title, credit=credit, page=page,
                            url=url, license="CC BY 4.0", license_url=LICENSE,
                            sha256=hashlib.sha256(blob).hexdigest(), shape=list(image.shape),
                            bytes=len(blob), use="unlabeled patch correspondence pretraining"))
        manifest_path.write_text(json.dumps(records, indent=2) + "\n")
        print("downloaded", identifier, image.shape, len(blob), flush=True)
    manifest_path.write_text(json.dumps(records, indent=2) + "\n")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("external/training_images"))
    download(parser.parse_args().output)
