import fitz, os
d = os.path.join(os.path.dirname(__file__), "report")
doc = fitz.open(os.path.join(d, "benchmark_summary.pdf"))
print("pages:", doc.page_count)
out = os.path.join(d, "_sum"); os.makedirs(out, exist_ok=True)
for i in range(doc.page_count):
    doc[i].get_pixmap(dpi=120).save(os.path.join(out, f"s{i+1}.png"))
print("rendered")
