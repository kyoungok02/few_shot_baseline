# few_shot_baseline

## Installation
### Environmetal setting
Set the environment using conda yaml file.     
```bash
conda env create --file conda/few_shot.yaml
```
Activate conda enviroment .  
```bash
conda activate few_shot </code>
```

### Dataset Download
Before running the code, it may be necessary to download the required datasets.  
The few shot datasets come from the link https://lyy.mpi-inf.mpg.de/mtl/download/.  
The dataset is loaded into the ```./data``` folder.  

## Train
This code can utilize two types of datasets: "mini-imagenet", "cifar100".  
For Prototypical Networks, the baseline network can be selected from two types of networks : "Baseline", "Resnet34".  
When using Visual Studio Code, you can utilize the following code in the json.  
For ProtoNet, use the ```train_protonet.py```.  
```javascript
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    // protoNet
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "args": ["--seed=7","--epoch=50", "--n_way=5", "--k_spt=5", 
            "--k_qry=15", "--imgsz=84", "--imgc=3","--lr=3e-2", "--batch_size=4","--resume=False",
            "--root_dir=$base_dir$", "--dataset=mini-imagenet",
            "--model=Baseline"]
            // imgsz = 84 for protoNet imgsz=105 for ResNet
        }
    ]
}
```
For SiameseNet, use the ```train_siamese.py```.  
```javascript
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    // protoNet
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
             "args": ["--epoch=30", "--n_way=1", "--k_spt=20", 
                 "--k_qry=20", "--imgsz=105", "--imgc=3", "--batch_size=4", "--lr=3e-2", "--resume=False",
                    "--root_dir=$base_dir$", "--dataset=mini-imagenet"
                 ]
                // imgsz = 84 for protoNet imgsz=105 for ResNet
        }
    ]
}
```
When you use the terminal, use the following command.  
For protonet,
```bash
cd $base_dir$
python train_protonet.py --seed=7 --epoch=50 --n_way=5 --k_spt=5 \\ 
            --k_qry=15 --imgsz=84 --imgc=3 --lr=3e-2 --batch_size=4 --resume=False \\
            --root_dir=$base_dir$ --dataset=mini-imagenet \\
            --model=Baseline
```

For SiameseNet,
```bash
cd $base_dir$
python train_siamese.py --epoch=30 --n_way=1 --k_spt=20 --k_qry=20 --imgsz=105 \\
                        --imgc=3 --batch_size=4 --lr=3e-2 --resume=False \\
                        --root_dir=$base_dir$ --dataset=mini-imagenet
```
