# FetchSlide_DDPG_HER
DDPG + HER implementation in PyTorch for FetchSlide Robot

## Instructions to run : 

1. Create a virtual environment and install required dependencies : 
pip3 install -r requirements.txt

2. To test using pretrained model : 
python main.py --phase=test

3. To train model using CPU :
  python main.py --phase=train

4. To train model using GPU (resolving bugs in gpu code): 
  python main.py --phase=train --cuda=True



