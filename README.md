# handwriting_transformer

## Installation
### Data
1. Download files from https://fki.tic.heia-fr.ch/databases/download-the-iam-on-line-handwriting-database into the `data/raw`.
2. Untar the files in `raw`

     `ls *.gz |xargs -n1 tar -xzf`
3. Run `python prepare_data.py` and the processed data will be in `data/processed`

## Data
1. Num of instance: 11615
2. Num of writers: 198


## TODO
1. change the loss function to be the mixture model
2. add noise and train
3. train with individual writer
4. decoding
5. add SOS and EOS to stroke sequence: current the end of sequence is [0,0,1] and it is not included
     1. <SOS> marks the start of a stroke
     2. <EOS> marks the end of a stroke
     3. <EOC> ends of character sequence
     4. <SOC> starts of character sequence
6. What is the value of the stroke when the pen is lifted?
7. At training time, set [0, 0, 1] to be the starting token. Add a end of stroke token to the end
8. At prediction time, the variable for pen lift should be 0 or 1
9. Does transformer starts training given the first token?


Diagnosis:
1. The prediction on stroke sequence is very wrong
     1. Either use absolute position of the stroke
     2. Incorporate conditioal depenendece when the pen is lifted up

People's suggestions:
1. fit a single batch
2. gradient clipping
3. use non continuous variables
