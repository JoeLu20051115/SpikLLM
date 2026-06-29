import argparse
import json

from bispikclm.data.fineweb import dataset_smoke_check, dataset_summary, prepare_dataset_manifests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap LM evaluation entrypoint.")
    parser.add_argument("--smoke-datasets", action="store_true", help="Validate dataset registry and manifest generation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke_datasets:
        manifest_dir = prepare_dataset_manifests()
        payload = {
            "summary": dataset_smoke_check(),
            "datasets": dataset_summary(),
            "manifest_dir": str(manifest_dir),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
