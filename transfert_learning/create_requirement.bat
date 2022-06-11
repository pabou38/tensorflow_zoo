
echo 'create requirement files'

:: if run from terminal, (type file name or double click), conda ,not found and requirement_conda is empty
rem go in conda gui, enable tf, create a powershell prompt,  and run from there

pip list --format=freeze > "my_requirement_pip.txt" 2>&1
:: pip freeze includes file path  --format=freeze is module=version 

conda list --export > "my_requirement_conda.txt" 2>&1
:: --export is module=version=source eg pypi

pip list --format=freeze > "my_requirement_pip.txt" 2>&1
:: pip freeze includes file path  --format=freeze is module=version 


rem WTF why is this not executed !!!!!!!!!!!!!  Looks like conda can only be executed once in bat file 

pip --version

conda env export > "env_export.yml" 2>&1
rem env export includes dependencies (conda modules ?) and pip
:: -n tf24   or will use current env


:: pip install -r requirements.txt
:: conda install --file requirements.txt

:: comment with rem will be printed. not with ::
