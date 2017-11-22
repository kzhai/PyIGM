import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import datetime, os;

import nltk;
import numpy;

import optparse;

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        #dataset_name=None,
                        
                        # parameter set 2
                        alpha_alpha=1,
                        alpha_kappa=1,
                        alpha_nu=1,
                        mu_0=None,
                        lambda_0=None,
                        
                        # parameter set 3
                        training_iterations=1000,
                        snapshot_interval=100,
                        #sampling_approach=0,
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
    #parser.add_option("--dataset_name", type="string", dest="dataset_name",
                      #help="the corpus name [None]");

    # parameter set 2
    parser.add_option("--alpha_alpha", type="float", dest="alpha_alpha",
                      help="hyper-parameter for Dirichlet process of cluster [1]")
    parser.add_option("--alpha_kappa", type="float", dest="alpha_kappa",
                      help="hyper-parameter for degree of freedom [1]")
    parser.add_option("--alpha_nu", type="float", dest="alpha_nu",
                      help="hyper-parameter for covariance matrix [1]")
    
    # parameter set 3
    parser.add_option("--training_iterations", type="int", dest="training_iterations",
                      help="number of training iterations [1000]");
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval [100]");
    #parser.add_option("--sampling_approach", type="int", dest="sampling_approach",
                      #help="sampling approach and heuristic [0(default):vanilla sampling, 1:component resampling, 2:random split-merge, 3:restricted split-merge]");
                      
    (options, args) = parser.parse_args();
    return options;

def main():
    options = parse_args();
    
    # parameter set 1
    #assert(options.dataset_name!=None);
    #dataset_name = options.dataset_name;
    
    assert(options.input_directory!=None);
    assert(options.output_directory!=None);
    
    #input_directory = options.input_directory;
    #input_directory = os.path.join(input_directory, dataset_name);
    
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
    input_file_path = os.path.join(input_directory, 'train.dat')
    train_data = numpy.loadtxt(input_file_path);
    print "successfully load all training data..."
    
    # parameter set 2
    assert options.alpha_alpha>0;
    alpha_alpha = options.alpha_alpha;
    assert options.alpha_kappa>0;
    alpha_kappa = options.alpha_kappa;
    assert options.alpha_nu>0;
    alpha_nu=options.alpha_nu;
    
    # parameter set 3
    if options.training_iterations>0:
        training_iterations=options.training_iterations;
    if options.snapshot_interval>0:
        snapshot_interval=options.snapshot_interval;
    #sampling_approach = options.sampling_approach;
    
    # create output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%m%d-%H%M%S")+"";
    suffix += "-%s" % ("igm");
    suffix += "-I%d" % (training_iterations);
    suffix += "-S%d" % (snapshot_interval);
    suffix += "-aa%g" % (alpha_alpha);
    suffix += "-ak%g" % (alpha_kappa);
    suffix += "-an%g" % (alpha_nu);
    #suffix += "-SA%d" % (sampling_approach);
    suffix += "/";

    output_directory = os.path.join(output_directory, suffix);
    os.mkdir(os.path.abspath(output_directory));
    
    # store all the options to a file
    options_output_file = open(output_directory + "option.txt", 'w');
    # parameter set 1
    options_output_file.write("input_directory=" + input_directory + "\n");
    options_output_file.write("dataset_name=" + dataset_name + "\n");
    # parameter set 2
    options_output_file.write("alpha_alpha=" + str(alpha_alpha) + "\n");
    options_output_file.write("alpha_nu=" + str(alpha_nu) + "\n");
    options_output_file.write("alpha_kappa=" + str(alpha_kappa) + "\n");
    # parameter set 3
    options_output_file.write("training_iteration=%d\n" % training_iterations);
    options_output_file.write("snapshot_interval=%d\n" % snapshot_interval);
    #options_output_file.write("sampling_approach=%d\n" % sampling_approach);
    options_output_file.close()
    
    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "output_directory=" + output_directory
    print "input_directory=" + input_directory
    print "dataset_name=" + dataset_name
    # parameter set 2
    print "alpha_alpha=" + str(alpha_alpha)
    print "alpha_nu=" + str(alpha_nu)
    print "alpha_kappa=" + str(alpha_kappa)
    # parameter set 3
    print "training_iteration=%d" % (training_iterations);
    print "snapshot_interval=%d" % (snapshot_interval);
    #print "sampling_approach=%d" % (sampling_approach)
    print "========== ========== ========== ========== =========="
    
    import monte_carlo;
    igm = monte_carlo.MonteCarlo();
    igm._initialize(train_data, alpha_alpha, alpha_kappa, alpha_nu);
    
    igm.export_snapshot(output_directory);
    
    for iteration in xrange(training_iterations):
        clock = time.time();
        log_likelihood = igm.learning();
        clock = time.time()-clock;
        print 'training iteration %d finished in %f seconds: number-of-clusters = %d, log-likelihood = %f' % (igm._iteration_counter, clock, igm._K, log_likelihood);

        if (igm._iteration_counter % snapshot_interval == 0):
            igm.export_snapshot(output_directory);

    igm.export_snapshot(output_directory);

if __name__ == '__main__':
    main()