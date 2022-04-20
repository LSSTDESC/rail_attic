jupyter nbconvert --ExecutePreprocessor.kernel_name=python --ExecutePreprocessor.timeout=600 --to html --execute examples/core/iterator_test.ipynb; \mv examples/core/iterator_test.html docs
jupyter nbconvert --ExecutePreprocessor.kernel_name=python --ExecutePreprocessor.timeout=600 --to html --execute examples/core/Pipe_Example.ipynb; \mv examples/core/Pipe_Example.html docs
jupyter nbconvert --ExecutePreprocessor.kernel_name=python --ExecutePreprocessor.timeout=600 --to html --execute examples/core/Run_Pipe.ipynb; \mv examples/core/Run_Pipe.html docs
jupyter nbconvert --ExecutePreprocessor.kernel_name=python --ExecutePreprocessor.timeout=600 --to html --execute examples/creation/degradation-demo.ipynb; \mv examples/creation/degradation-demo.html docs
jupyter nbconvert --ExecutePreprocessor.kernel_name=python --ExecutePreprocessor.timeout=600 --to html --execute examples/creation/posterior-demo.ipynb; \mv examples/creation/posterior-demo.html docs
jupyter nbconvert --ExecutePreprocessor.kernel_name=python --ExecutePreprocessor.timeout=600 --to html --execute examples/estimation/RAIL_estimation_demo.ipynb; \mv examples/estimation/RAIL_estimation_demo.html docs
jupyter nbconvert --ExecutePreprocessor.kernel_name=python --ExecutePreprocessor.timeout=600 --to html --execute examples/evaluation/demo.ipynb; \mv examples/evaluation/demo.html docs
jupyter nbconvert --ExecutePreprocessor.kernel_name=python --ExecutePreprocessor.timeout=600 --to html --execute examples/goldenspike/goldenspike.ipynb; \mv examples/goldenspike/goldenspike.html docs
