"""Generate a small multi-page PDF for ingestion testing.

Uses pypdf only (already a dependency) — no extra packages.
"""

from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    RectangleObject,
    TextStringObject,
)


def _content_stream(lines: list[str]) -> DecodedStreamObject:
    body = "BT\n/F1 12 Tf\n72 720 Td\n14 TL\n"
    body += "".join(f"({line}) Tj T*\n" for line in lines)
    body += "ET"
    stream = DecodedStreamObject()
    stream.set_data(body.encode("latin-1"))
    return stream


def make_pdf(path: Path) -> None:
    writer = PdfWriter()

    pages_text = [
        [
            "Project Atlas: Technical Overview",
            "",
            "Project Atlas is an internal data-platform initiative kicked off in",
            "March 2026. The platform unifies telemetry from three subsystems:",
            "ingestion, storage, and query routing. It targets sub-second p99",
            "latency for analytical queries on datasets up to 10 TB.",
            "",
            "The lead architect is Priya Raman. Engineering is split across two",
            "pods: Pod Echo handles ingestion and Pod Foxtrot handles query.",
        ],
        [
            "Architecture and Storage",
            "",
            "Atlas uses a columnar storage layer based on Apache Parquet, with",
            "a metadata service backed by FoundationDB. Compaction runs every",
            "six hours and emits Prometheus metrics on the namespace",
            "atlas.compaction.*.",
            "",
            "Hot data lives in NVMe-backed nodes; cold data tiers to S3 after",
            "30 days. Replication factor is 3 in the hot tier.",
        ],
        [
            "Operational Notes",
            "",
            "On-call rotation is weekly. The runbook lives at docs/atlas/runbook.md.",
            "Page severity is set by the SEV field in PagerDuty: SEV1 pages the",
            "primary oncall and the architect; SEV3 only emails the team list.",
            "",
            "Known issue: query routing occasionally misroutes when a tenant has",
            "more than 2,000 partitions. Workaround is documented in JIRA-4421.",
        ],
    ]

    for lines in pages_text:
        page = writer.add_blank_page(width=612, height=792)
        content = _content_stream(lines)
        page[NameObject("/Contents")] = content
        page[NameObject("/MediaBox")] = RectangleObject([0, 0, 612, 792])

        font_dict = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        resources = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font_dict}
                ),
            }
        )
        page[NameObject("/Resources")] = resources

    with open(path, "wb") as f:
        writer.write(f)


if __name__ == "__main__":
    out = Path(__file__).parent / "project_atlas.pdf"
    make_pdf(out)
    print(f"Wrote {out}")
