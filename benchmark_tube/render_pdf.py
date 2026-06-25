import fitz, os
d = os.path.join(os.path.dirname(__file__), "report")
doc = fitz.open(os.path.join(d, "benchmark_report.pdf"))
print("pages:", doc.page_count)
out = os.path.join(d, "_pg"); os.makedirs(out, exist_ok=True)
for i in range(doc.page_count):
    doc[i].get_pixmap(dpi=110).save(os.path.join(out, f"p{i+1}.png"))
print("rendered to", out)
