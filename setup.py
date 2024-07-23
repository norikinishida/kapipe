# Copyright 2024 Noriki Nishida. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from setuptools import setup, find_packages
import shutil


if __name__ == "__main__":
    python_requires = ">=3.10"
    install_requires = [
        "numpy>=1.22.2",
        "scipy>=1.10.1",
        "pandas>=1.5.3",
        "spacy>=3.7.1",
        "torch",
        "opt-einsum>=3.3.0",
        "faiss-gpu>=1.7.2",
        "pyhocon>=0.3.60",
        "tqdm>=4.66.1",
        "jsonlines>=4.0.0",
        "Levenshtein>=0.25.0",
        "transformers",
        "accelerate",
        "bitsandbytes"
    ]
    setup(
        name="kapipe",
        version="0.0.3",
        author= "Noriki Nishida",
        author_email="norikinishida@gmail.com",
        description="A learnable pipeline for knowledge acquisition",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords="NLP, knowledge acquisition, information extraction, named entity recognition, entity disambiguation, relation extraction",
        license="Apache License 2.0",
        url="https://github.com/norikinishida/kapipe",
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
        python_requires=python_requires,
        install_requires=install_requires,
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )

