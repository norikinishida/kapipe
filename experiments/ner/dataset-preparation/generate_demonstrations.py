import argparse
import os

from kapipe import utils


def main(args: argparse.Namespace) -> None:
    # Load candidate demonstration documents
    documents: list[dict] = utils.read_json(args.input_file)

    # Sort documents by the number of gold mentions
    sorted_documents = sorted(
        documents,
        key=lambda document: -len(document["mentions"]),
    )

    # Select top-k documents as fixed few-shot demonstrations
    demonstration_documents = sorted_documents[:args.n_demonstrations]

    # Create the output directory
    utils.mkdir(os.path.dirname(args.output_file))

    # Save documents
    utils.write_json(args.output_file, demonstration_documents)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--n_demonstrations", type=int, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    main(args=args)