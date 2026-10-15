"""Write the packaged viewer to viewer.html, or check that it is current.

    python -m streetview_to_3d.visualise
    python -m streetview_to_3d.visualise --check
"""
import argparse

from streetview_to_3d.visualise.build_viewer import OUTPUT, build_viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if viewer.html is stale")
    args = parser.parse_args()
    document = build_viewer()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != document:
            raise SystemExit("viewer.html is stale. Run python -m streetview_to_3d.visualise")
        print("viewer.html matches its modular source")
    else:
        OUTPUT.write_text(document, encoding="utf-8")
        print(f"Built {OUTPUT}")


if __name__ == "__main__":
    main()
