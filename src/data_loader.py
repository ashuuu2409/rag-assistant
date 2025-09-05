# src/data_loader.py
import json

def load_publications(json_path="data/project_1_publications.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pubs = []
    for pub in data:
        title = pub.get("title", "Untitled")
        content = pub.get("publication_description", "")
        if content:
            pubs.append((title, content))
    return pubs
