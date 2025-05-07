import re
def build_metadata_blurb(records: list[dict]) -> list[dict]:
    """
    group "high signal" data points (e.g. hed/summary) - typeOfMaterials is always "news"
    """
    blurbs = []
    for rec in records: 
        raw_tags = rec.get("timesTags", [])
        if isinstance(raw_tags, (list, set, tuple)):
            tags_txt = ", ".join(map(str, raw_tags))
        else:
            tags_txt = str(raw_tags)

        key_text =  [
            f"Headline: {rec.get("headline", "")}",
            f"By: "+ rec.get("bylines", ""),
            f"Tone: {rec.get('tone', '')}".strip(),
            f"Section: {rec.get('typeOfMaterials', [''])[0].strip()}",
            f"Published: {rec.get("firstPublished", "")[:10].strip()}", 
            f"Tags: {tags_txt}",
            f"Summary: {rec.get("summary", "")}"
        ]

        # print(key_text)
        blurb = " ".join(filter(None, key_text))
        blurb = re.sub(r"\s+", " ", blurb).strip()
        blurbs.append(blurb)
    return blurbs