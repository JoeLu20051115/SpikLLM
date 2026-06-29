import argparse
import json

from bispikclm.data.fineweb import download_teachers, prepare_dataset_manifests
from bispikclm.distill.spad import SpADConfig, summarize_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap SPAD training entrypoint.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved bootstrap plan.")
    parser.add_argument("--download-teachers", action="store_true", help="Cache teacher metadata and tokenizer assets.")
    parser.add_argument("--prepare-datasets", action="store_true", help="Write dataset manifests for smoke runs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = {"plan": summarize_plan(SpADConfig())}
    if args.download_teachers:
        payload["teachers"] = download_teachers()
    if args.prepare_datasets:
        payload["dataset_manifest_dir"] = str(prepare_dataset_manifests())
    if args.dry_run or args.download_teachers or args.prepare_datasets:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

