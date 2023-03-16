# few_shot_baseline

## Installation
### Environmetal setting
Set the environment using conda yaml file.     
<code> conda env create --file conda/few_shot.yaml </code>   
Activate conda enviroment .  
<code> conda activate few_shot </code>.  

### Dataset Download
Before running the code, it may be necessary to download the required datasets.  
The few shot datasets come from the link https://lyy.mpi-inf.mpg.de/mtl/download/.  
The dataset is loaded into the ```./data``` folder.  

## Train
When using Visual Studio Code, you can utilize the following code in the json.  
For ProtoNet,
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
            "--root_dir=/Users/kyoung-okyang/few_shot/few_shot_baseline_1", "--dataset=mini-imagenet",
            "--model=Baseline"]
            // imgsz = 84 for protoNet imgsz=105 for ResNet
        }
    ]
}
```
For SiameseNet.  
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
                    "--root_dir=/Users/kyoung-okyang/few_shot/few_shot_baseline_1", "--dataset=mini-imagenet"
                 ]
                // imgsz = 84 for protoNet imgsz=105 for ResNet
        }
    ]
}
```
