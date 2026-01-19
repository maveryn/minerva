import argparse
from pathlib import Path

from detection.build_art import build_art_dataset
from detection.build_elastic import build_elastic_dataset
from detection.build_sentinel import build_sentinel_dataset
from detection.build_splunk import build_splunk_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all detection datasets.")
    parser.add_argument("--output-dir", default="dataset/detection")
    parser.add_argument("--cache-dir", default="dataset/detection_cache")
    parser.add_argument("--mitre-path", default="dataset/mitre/enterprise-attack.json")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--with-distractors", action="store_true")
    parser.add_argument("--distractor-count", type=int, default=6)
    parser.add_argument("--include-names", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    mitre_path = Path(args.mitre_path)

    build_art_dataset(
        output_dir=output_dir,
        cache_dir=cache_dir,
        mitre_path=mitre_path,
        seed=args.seed,
        with_distractors=args.with_distractors,
        distractor_count=args.distractor_count,
        include_names=args.include_names,
    )
    build_sentinel_dataset(
        output_dir=output_dir,
        cache_dir=cache_dir,
        mitre_path=mitre_path,
        seed=args.seed,
        with_distractors=args.with_distractors,
        distractor_count=args.distractor_count,
        include_names=args.include_names,
    )
    build_splunk_dataset(
        output_dir=output_dir,
        cache_dir=cache_dir,
        mitre_path=mitre_path,
        seed=args.seed,
        with_distractors=args.with_distractors,
        distractor_count=args.distractor_count,
        include_names=args.include_names,
    )
    build_elastic_dataset(
        output_dir=output_dir,
        cache_dir=cache_dir,
        mitre_path=mitre_path,
        seed=args.seed,
        with_distractors=args.with_distractors,
        distractor_count=args.distractor_count,
        include_names=args.include_names,
    )

    print(f"Wrote detection datasets to {output_dir}")


if __name__ == "__main__":
    main()
