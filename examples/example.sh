#!/usr/bin/env sh


python ./example_pipeline.py --document ./example-documents/example1.json
# python ./example_pipeline.py --document ./example-documents/example2.json

# python ./example_specific_task.py --document ./example-documents/example1.json
# python ./example_specific_task.py --document ./example-documents/example2.json

# python ./example_train.py \
#     --train_documents ./example-documents/cdr_train.json \
#     --dev_documents ./example-documents/cdr_train.json \
#     --entity_dict ${HOME}/.kapipe/data/kb/mesh/mesh2015.entity_dict.json

