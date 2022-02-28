\rm -rf rail/estimation/data/AB/E*.AB rail/estimation/data/AB/I*.AB rail/estimation/data/AB/S*.AB rail/estimation/data/AB/s*.AB
python -m pytest --cov=./rail --cov-report=html tests
