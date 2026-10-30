# Working with this pipeline
1. Make sure the file`params.json` has all the original parameters required to run the test.
2. Open the file `make_json.py` and make sure test parameter values are present and modify them if required. This script removes old .json files and creates new ones according to the test parameters mentioned.  
3. Now run the file `robustness.py`. This file removes any numpy arrays previously stored in the directories and replaces them with the returns of the robustness test.
4. Once the robustness is completed. Open the Jupyter Notebook `analysis.ipynb` and go thru each cells to do a performance analysis of for each parameter.  