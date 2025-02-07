setup:
	python3 -m venv ~/.ML_project

install: 
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest --nbval ML_project/10681109_Cantore_Andrea-2.ipynb

format: 
	black *.py # cleans up the code


all: install test 
