# AMINE_Multiview (Active Module Identification through Network Embedding with multiview)

## Installation

Clone the repository and change to the project directory
```bash
git clone https://github.com/tchotaneu/Intership_I3S.git

cd amine
```

To set up the necessary dependencies for the project, you can create a virtual environment using your preferred package manager. In this example, we will demonstrate using mamba, a more efficient alternative to conda. The following commands will create the environment and install all dependencies:

```bash
# Using mamba
mamba env create -n amine-env -f environment.yml
mamba activate amine-env
```

Mamba is recommended over conda for this task, especially for larger installations, as it typically performs the installation significantly faster than conda, especially versions older than 23.1.0, which may take over 10 minutes.

Alternatively, you can manually create the environment and install dependencies using conda with the following commands:

```bash
# Using conda
conda create -n amine-env python=3.6
conda activate amine-env
conda install -c conda-forge -c anaconda -c numba networkx scipy gensim numba pandas xlrd scikit-learn powerlaw progressbar2 openpyxl python-levenshtein pyyaml
```

Additionally, you have the option to create a virtual environment using the built-in Python venv module. The specific commands depend on your operating system:

- For Linux or macOS :

```bash
python3 -m venv amine-env
source amine-env/bin/activate
```

- For Windows:

```bash
python -m venv amine-env
.\amine-env\Scripts\activate
```
Choose the method that best suits your operating system. 

At this point, you can download network files with the following command:
```bash
python -m amine.download_data
```
that will download data in the ./data/real/ppi directory

## networks
The downloaded protein-protein interaction networks originating from 3 different databases and concerning 3 different species are described below.

### from STRING database (https://string-db.org/)
* human: network of 19,382 nodes and 5,968,679 edges
* mouse: 21,317 nodes and 7,248,179 edges
* drosophila: network of 13,047 nodes and 2,171,899 edges

### from IntAct database (https://www.ebi.ac.uk/intact)
* human: network of 17,721 nodes and 314,807 edges
* mouse: network of 6,998 nodes and 18,944 edges
* drosophila: network of 6,610 nodes and 29,305 edges

### from BioGRID database (https://thebiogrid.org/)
* human: network of 19,892 nodes and 780,328 edges
* mouse: network of 10,949 nodes and 60,254 edges
* drosophila: network of 9,514 nodes and 64,927 edges

### merged networks
In addition to the interactions networks stored in specific databases, we propose, for each specie, a network that contains a fusion of all interactions stored in Intact, BioGRID or STRING (for this database, only the interactions associated with a global score >= 0.7 are retained).
* for human, this represents a network with 21,936 nodes and 1,056,188 edges
* for mouse, this represents a network with 18,156 nodes and 283,649 edges
* for drosophila, this represents a network with 14,412 nodes and 211,618 edges

## Usage

### On real data
The program can be executed with the command:
```bash
python -m amine
```
The -h option displays explanations of the available options. A typical execution can be performed with:
```bash
python -m amine --expvalue ./data/real/expression/chiou_2017/Hmga2_positive_vs_negative.csv -g 0 -l 2 -p 6 -s mouse -n string -o ./data/results/Hmga2_positive_vs_negative_string_network.xlsx -v
```
The command above runs amine on the result of a differential expression analysis stored in the file specified with the parameter **--expvalue**. The parameters **-g**, **-l** and **-p** are used to specify the column with the gene names, the log2 fold changes and the p-values respectively. Numbering starts at 0, which means that zero identifies the first column. The parameter **-s** is used to indicate the specie and the parameter **-n** is used to indicate the origin of the interaction network (in the example, it is the STRING database). The parameter **-o** allows to specify the path to a file to write the results.

The file "execute_reals.sh" contains examples of commands to process data from the paper of Chiou 2017 using different networks.

Other parameters can be used to filter the network to be used by specifying them in the "config.yaml" file.

### On artificial data
The method can be executed in batch mode on set of artificially generated datasets with the command:
```bash
python -m amine.process_artificial_network
```
The -h option displays explanations of the available options. Below are two examples of commands to run the program on a batch of artificial data:

* running AMINE on the dataset of 1,000 networks with 3 modules of 10 nodes used in the Robinson et al. paper and saving the results in the file "./data/results/guyon.txt":
```bash
python -m amine.process_artificial_network -g guyondata -r 1000 -m 10 -n 3 -o ./data/results/guyon.txt -v
```
* executing AMINE on 100 dense networks of 1,000 nodes with 1 modules of 20 nodes and saving the results in the file "./data/results/dense_1Knodes_moduleof20.txt":
```bash
python -m amine.process_artificial_network -r 100 -m 20 -s 1000 -o ./data/results/dense_1Knodes_moduleof20.txt -v
```
