import argparse
import json
import os

import sys
sys.path.insert(0, "../../..")
from kapipe import utils


def main(args):
    path_input_dir = args.input_dir
    path_output_file = args.output_file

    utils.mkdir(os.path.dirname(path_output_file))

    with open(path_output_file, "w") as out_file:
        dir_names = os.listdir(path_input_dir)
        dir_names = sorted(dir_names)
        for dir_name in dir_names:
            print(f"Processing {path_input_dir}/{dir_name} ...")
            filenames = os.listdir(os.path.join(path_input_dir, dir_name))
            filenames = sorted(filenames)
            for i, filename in enumerate(filenames):
                print(f"\t[{i+1}/{len(filenames)}] Processing {path_input_dir}/{dir_name}/{filename} ...")
                with open(os.path.join(path_input_dir, dir_name, filename)) as in_file:
                    for line in in_file:
                        article = json.loads(line)
                        # Exclude redirect page
                        if is_redirect(article=article):
                            continue
                        # article = process_article(article)
                        json_str = json.dumps(article, ensure_ascii=True)
                        out_file.write(json_str + "\n")
   

def is_redirect(article):
    text = article["text"].strip()
    if not text:
        return True
    elif text.lower().startswith("#redirect"):
        return True
    elif len(text.replace("\n", " ").split(" ")) <= 10:
        return True
    else:
        return False


# def process_article(article):
#     # Process title, i.e., replacing " " with "_"
#     title = article["title"]
#     title = title.replace(" ", "_")
#     article["title"] = title
#     return article

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)





