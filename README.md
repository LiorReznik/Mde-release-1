# Mde-release-1
Automated Discovery of Mathematical Definitions in Text
instructions:
1.	Install all the dependencies:
    - python 3.6.x and all the packages:
      - Upgrade pip by executing the following command: python -m pip install --upgrade pip 
      - Install pytorch: pip install -U setuptools==49.2.0 torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
      - Install all the rest: pip install -r PATHTOROOTFOLDER/requirements.txt
    - Install java runtime environment:
      - Ubuntu: sudo apt update , sudo apt install default-jre
      - Windows: https://www.oracle.com/java/technologies/javase-jre8-downloads.html
2. To communicate with the software, you need to use the CL interface, there are 3 arguments to enter:
   - -i (--input_path) a path to the file that needs to be annotated.
   - (optional) -m (--model) a name of the desired model; the choices are: cnn_mld (default), cblstm_ml, cblstm_mld.
   - (optional) -o (--output_path) the path and name of the output file (default: ./data_annoated.txt)
3. commands for example, using our demo file (assuming you are in the projects root folder):
   - python ./main.py -i ./demo.txt -m cblstm_ml -o ./res.txt (applies CBLSTM_ml model on demo.txt and write the result into the res.txt in the -current directory)
   - python ./main.py -i ./demo.txt -m cblstm_mld -o ./res2.txt  
   - python ./main.py -i ./demo.txt 
   
   
Please note:
- At the first run of the script, the script downloads resources from the net, so it will take some time to finish.
- If your default python is not python 3 you may need to use pip3 and python3 in your commands.
- The work is licensed under the MIT licence, for more details: https://github.com/LiorReznik/Mde-release-1/blob/master/LICENSE
