PyIGM
==========

PyIGM is a Infinite Gaussian Mixture package, developed by the Cloud Computing Research Team in [University of Maryland, College Park] (http://www.umd.edu).

Please download the latest version from our [GitHub repository](https://github.com/kzhai/PyIGM).

Please send any bugs of problems to Ke Zhai (kzhai@umd.edu).

Install and Build
----------

This package depends on many external python libraries, such as numpy, scipy and nltk.

Launch and Execute
----------

Assume the PyIGM package is downloaded under directory ```$PROJECT_SPACE/src/```, i.e., 

	$PROJECT_SPACE/src/PyIGM

To prepare the example dataset,

	tar zxvf point-clusters.tar.gz

To launch PyIGM, first redirect to the parent directory of PyIGM source code,

	cd $PROJECT_SPACE/src/

and run the following command on example dataset,

	python -m PyIGM.launch_train --input_directory=./PyIGM/point-clusters --output_directory=./PyIGM/ --training_iterations=100

The generic argument to run PyIGM is

	python -m PyIGM.launch_train --input_directory=$INPUT_DIRECTORY/$DATASET_NAME --output_directory=$OUTPUT_DIRECTORY --training_iterations=$NUMBER_OF_ITERATIONS

You should be able to find the output at directory ```$OUTPUT_DIRECTORY/$DATASET_NAME```.

Under any cirsumstances, you may also get help information and usage hints by running the following command

	python -m PyIGM.launch_train --help
