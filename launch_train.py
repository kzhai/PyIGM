#!/usr/bin/python
import cPickle, getopt, sys, time, re
import datetime, os;

import scipy.io;
import nltk;
import numpy;
import optparse;

"""
normalize the data, i.e., subtract the mean, and divide by the variance
"""
def normalize_data(data):
    (N, D) = data.shape;
    data = data - numpy.tile(data.mean(axis=0), (N, 1));
    data = data / numpy.var(data, axis=0)[numpy.newaxis, :];
    return data

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        # dictionary=None,
                        
                        # parameter set 2
                        training_iterations=-1,
                        snapshot_interval=10,
                        # number_of_topics=-1,

                        # parameter set 3
                        alpha_alpha=1,
                        initial_kappa=1,
                        initial_nu=1,
                        
                        # parameter set 4
                        # disable_alpha_theta_update=False,
                        inference_mode=0,
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
    # parser.add_option("--corpus_name", type="string", dest="corpus_name",
                      # help="the corpus name [None]")
    # parser.add_option("--dictionary", type="string", dest="dictionary",
                      # help="the dictionary file [None]")
    
    # parameter set 2
    parser.add_option("--training_iterations", type="int", dest="training_iterations",
                      help="total number of iterations [-1]");
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval [10]");                      
                      
    # parameter set 3
    parser.add_option("--alpha_alpha", type="float", dest="alpha_alpha",
                      help="Dirichlet process scale parameter [1.0]")
    parser.add_option("--initial_kappa", type="float", dest="initial_kappa",
                      help="mean fraction [1.0]")
    parser.add_option("--initial_nu", type="float", dest="initial_nu",
                      help="degree of freedom [1.0]")
    
    # parameter set 4
    # parser.add_option("--disable_alpha_theta_update", action="store_true", dest="disable_alpha_theta_update",
                      # help="disable alpha_alpha (hyper-parameter for Dirichlet distribution of topics) update");
    # parser.add_option("--inference_mode", type="int", dest="inference_mode",
                      # help="inference mode [ " + 
                            # "0 (default): hybrid inference, " + 
                            # "1: monte carlo, " + 
                            # "2: variational bayes " + 
                            # "]");
    # parser.add_option("--inference_mode", action="store_true", dest="inference_mode",
    #                  help="run latent Dirichlet allocation in lda mode");

    (options, args) = parser.parse_args();
    return options;

def main():
    options = parse_args();

    # parameter set 2
    # assert(options.number_of_topics>0);
    # number_of_topics = options.number_of_topics;
    assert(options.training_iterations > 0);
    training_iterations = options.training_iterations;
    assert(options.snapshot_interval > 0);
    if options.snapshot_interval > 0:
        snapshot_interval = options.snapshot_interval;
    
    # parameter set 4
    # disable_alpha_theta_update = options.disable_alpha_theta_update;
    # inference_mode = options.inference_mode;
    
    # parameter set 1
    # assert(options.dataset_name!=None);
    assert(options.input_directory != None);
    assert(options.output_directory != None);
    
    input_directory = options.input_directory;
    input_directory = input_directory.rstrip("/");
    dataset_name = os.path.basename(input_directory);
    
    output_directory = options.output_directory;
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    output_directory = os.path.join(output_directory, dataset_name);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    
    # Dataset
    train_file_path = os.path.join(input_directory, 'train.dat')
    train_data = numpy.loadtxt(train_file_path)
    train_data = normalize_data(train_data);
    print "successfully load all train_data from %s..." % (os.path.abspath(train_file_path));
    
    # parameter set 3
    assert options.alpha_alpha > 0
    alpha_alpha = options.alpha_alpha;
    assert options.initial_kappa > 0;
    initial_kappa = options.initial_kappa;
    assert options.initial_nu > 0;
    initial_nu = options.initial_nu;

    # create output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%m%d-%H%M%S") + "";
    suffix += "-%s" % ("lda");
    suffix += "-I%d" % (training_iterations);
    suffix += "-S%d" % (snapshot_interval);
    # suffix += "-K%d" % (number_of_topics);
    suffix += "-aa%f" % (alpha_alpha);
    suffix += "-kz%f" % (initial_kappa);
    suffix += "-nz%f" % (initial_nu);
    # suffix += "-im%d" % (inference_mode);
    # suffix += "-%s" % (resample_topics);
    # suffix += "-%s" % (hash_oov_words);
    suffix += "/";
    
    output_directory = os.path.join(output_directory, suffix);
    os.mkdir(os.path.abspath(output_directory));

    # dict_file = options.dictionary;
    # if dict_file != None:
        # dict_file = dict_file.strip();
        
    # store all the options to a file
    options_output_file = open(output_directory + "option.txt", 'w');
    # parameter set 1
    options_output_file.write("input_directory=" + input_directory + "\n");
    options_output_file.write("dataset_name=" + dataset_name + "\n");
    # options_output_file.write("vocabulary_path=" + str(dict_file) + "\n");
    # parameter set 2
    options_output_file.write("training_iterations=%d\n" % (training_iterations));
    options_output_file.write("snapshot_interval=" + str(snapshot_interval) + "\n");
    # options_output_file.write("number_of_topics=" + str(number_of_topics) + "\n");
    # parameter set 3
    options_output_file.write("alpha_alpha=" + str(alpha_alpha) + "\n");
    options_output_file.write("initial_kappa=" + str(initial_kappa) + "\n");
    options_output_file.write("initial_nu=" + str(initial_nu) + "\n");
    # parameter set 4
    # options_output_file.write("inference_mode=%d\n" % (inference_mode));
    options_output_file.close()

    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "output_directory=" + output_directory
    print "input_directory=" + input_directory
    print "dataset_name=" + dataset_name
    # print "dictionary file=" + str(dict_file)
    # parameter set 2
    print "training_iterations=%d" % (training_iterations);
    print "snapshot_interval=" + str(snapshot_interval);
    # print "number_of_topics=" + str(number_of_topics)
    # parameter set 3
    print "alpha_alpha=" + str(alpha_alpha)
    print "initial_kappa=" + str(initial_kappa)
    print "initial_nu=" + str(initial_nu)
    # parameter set 4
    # print "inference_mode=%d" % (inference_mode)
    print "========== ========== ========== ========== =========="
    
    import monte_carlo;
    igm_inferencer = monte_carlo.MonteCarlo();
    igm_inferencer._initialize(train_data, alpha_alpha, initial_kappa, initial_nu);
    
    for iteration in xrange(training_iterations):
        log_likelihood = igm_inferencer.learning();
        
        print "iteration: %i\tK: %i\tlikelihood: %f" % (igm_inferencer._counter, igm_inferencer._K, log_likelihood);

        if (igm_inferencer._counter % snapshot_interval == 0):
            igm_inferencer.export_snapshot(output_directory);
            print "successfully export the snapshot to " + output_directory + " for iteration " + str(igm_inferencer._counter) + "..."
        
        print igm_inferencer._K
        print igm_inferencer._count[:igm_inferencer._K]
        # print igm_inferencer._label
    
    model_snapshot_path = os.path.join(output_directory, 'model-' + str(igm_inferencer._counter));
    cPickle.dump(igm_inferencer, open(model_snapshot_path, 'wb'));
    
if __name__ == '__main__':
    main()
