# handwriting_transformer

## Installation
### Data
1. Download files from https://fki.tic.heia-fr.ch/databases/download-the-iam-on-line-handwriting-database into the `data/raw`.
2. Untar the files in `raw`

     `ls *.gz |xargs -n1 tar -xzf`
3. Run `python prepare_data.py` and the processed data will be in `data/processed`

![Read Paper Here](advanced_deep_learning_final_project.pdf "")
