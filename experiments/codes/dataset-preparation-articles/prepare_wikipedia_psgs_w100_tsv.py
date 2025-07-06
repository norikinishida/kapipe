import argparse
import csv
import json
import os

import sys
sys.path.insert(0, "../../..")
from kapipe import utils


def main(args):
    path_input_file = args.input_file
    path_output_file = args.output_file

    utils.mkdir(os.path.dirname(path_output_file))

    with open(path_output_file, "w") as out_file,\
         open(path_input_file) as in_file:
        reader = csv.reader(in_file, delimiter="\t")
        for k, row in enumerate(reader):
            if not row[0] == "id":
                passage = {
                    "title": row[2],
                    "text": row[1],
                    "passage_id": row[0],
                }
                json_str = json.dumps(passage)
                out_file.write(json_str + "\n")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)





