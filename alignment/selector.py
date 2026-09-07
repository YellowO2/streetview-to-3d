"""The segment selector: pick which pieces to run road alignment on.

Part of the pipeline rather than a test, because choosing the input is a
step in running it. It was in tests/ only because that is where it was
first thrown together.

The page used to be produced by a script in /tmp, so when it needed a
change there was nothing to rerun and the generated file had to be edited
by hand -- which then no longer matched the template it came from. The
node data lived only inside that HTML as well. Now the data is in data/
and the page is regenerated from alignment.viz, so the template stays the
single source and the HTML is a build artefact.

    python -m alignment.selector
    python -m http.server -d build 8000
"""
import os
from alignment import viz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "selector_nodes.json")
OUT = os.path.join(ROOT, "build", "gap_selector.html")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    viz.build(
        data_path=DATA,
        pos_field="pos",
        ref_field="ref",
        group_field="group",
        label_field="label",
        residual_field="residual_m",
        chunk_field="chunk_id",
        title="NTU segments",
        subtitle="Pick the segments to run road alignment on. "
                 "The selected list is copy-pasteable.",
        out_path=OUT,
        selectable=True,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
