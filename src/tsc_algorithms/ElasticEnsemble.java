package tsc_algorithms;

import fileIO.OutFile;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.lazy.kNN;
import weka.filters.timeseries.DerivativeFilter;
import weka.core.elastic_distance_measures.BasicDTW;
import weka.core.elastic_distance_measures.ERPDistance;
import weka.core.elastic_distance_measures.LCSSDistance;
import weka.core.elastic_distance_measures.MSMDistance;
import weka.core.elastic_distance_measures.SakoeChibaDTW;
import weka.core.elastic_distance_measures.TWEDistance;
import weka.core.elastic_distance_measures.WeightedDTW;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import weka.classifiers.Classifier;
import weka.core.EuclideanDistance;
import utilities.ClassifierTools;

public class ElasticEnsemble implements Classifier {
	double[] predictions;

	// note distributionForInstance and getCapabilities added to appease the new
	// Classifier interface, NO IMPLEMENTATION
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		throw new UnsupportedOperationException("Not supported yet.");
	}

	@Override
	public Capabilities getCapabilities() {
		throw new UnsupportedOperationException("Not supported yet.");
	}

	public enum ClassifierVariants {
		Euclidean_1NN, DTW_R1_1NN, DTW_Rn_1NN, WDTW_1NN, DDTW_R1_1NN, DDTW_Rn_1NN, WDDTW_1NN, LCSS_1NN, MSM_1NN, TWE_1NN, ERP_1NN,
	}
	
	public enum EnsembleType {
		Best, Equal, Prop, Signif
	}

	protected static double[] msmParms = {
			// <editor-fold defaultstate="collapsed" desc="hidden for space">
			0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325, 0.03625, 0.04, 0.04375, 0.0475, 0.05125, 0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375, 0.0775, 0.08125, 0.085, 0.08875, 0.0925, 0.09625, 0.1, 0.136, 0.172, 0.208, 0.244, 0.28, 0.316, 0.352, 0.388, 0.424, 0.46, 0.496, 0.532, 0.568, 0.604, 0.64, 0.676, 0.712, 0.748, 0.784, 0.82, 0.856, 0.892, 0.928, 0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8, 3.16, 3.52, 3.88, 4.24, 4.6, 4.96, 5.32, 5.68, 6.04, 6.4, 6.76, 7.12, 7.48, 7.84, 8.2, 8.56, 8.92, 9.28, 9.64, 10, 13.6, 17.2, 20.8, 24.4, 28, 31.6, 35.2, 38.8, 42.4, 46, 49.6, 53.2, 56.8, 60.4, 64, 67.6, 71.2, 74.8, 78.4, 82, 85.6, 89.2, 92.8, 96.4, 100 // </editor-fold>
	};

	protected static double[] twe_nuParams = {
			// <editor-fold defaultstate="collapsed" desc="hidden for space">
			0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1,// </editor-fold>
	};

	protected static double[] twe_lamdaParams = {
			// <editor-fold defaultstate="collapsed" desc="hidden for space">
			0, 0.011111111, 0.022222222, 0.033333333, 0.044444444, 0.055555556, 0.066666667, 0.077777778, 0.088888889, 0.1,// </editor-fold>
	};

	// Initially uses a TreeSet to store the classifiers to use. This ensures
	// that no duplicates are included, and keeps everything in the expected
	// order as specified in the enum creation.
	// Once classifier is built, the classifier choices are locked in by
	// creating an array of ClassifierVariants. This is done for two reasons;
	// firstly, the index of the array matches
	// the other arrays, such as cvAccs and cvPreds. Secondly, this seperates
	// the classifier selection before and after building the classifier,
	// ensuring that unexpected behaviour isn't caused
	// by carrying out abnormal opperations (i.e. adding classifiers to the
	// ensemble after training has occured).
	protected TreeSet<ClassifierVariants> classifiersToUse;
	protected ClassifierVariants[] finalClassifiers;

	protected double[] cvAccs;
	protected double[][] cvPreds;
	protected double[] trainActualClassVals;

	protected EnsembleType ensembleType;
	protected double[][] bestParams;
	protected boolean fileWriting;

	protected String outpurDirLocation;
	protected String datasetName;

	protected Instances fullTrainingData;
	protected boolean[] mcNemarsInclusion;

	protected boolean classifierBuilt;
	protected boolean verbose;

	private boolean parallel;

	public double[] getCVAccs() {
		return cvAccs;
	}

	public double[][] getbestParams() {
		return bestParams;
	}

	public ElasticEnsemble() {
		setEnsembleType(ElasticEnsemble.EnsembleType.Prop);
		this.classifiersToUse = new TreeSet<>();
		classifiersToUse.addAll(Arrays.asList(ClassifierVariants.values()));

		this.finalClassifiers = null;
		this.fileWriting = false;
		this.outpurDirLocation = null;
		this.cvAccs = null;
		this.cvPreds = null;
		this.bestParams = null;
		this.verbose = false;
		this.classifierBuilt = false;
		this.parallel = true; // default
	}

	public void turnAllClassifiersOn() throws Exception {
		if (classifierBuilt) {
			throw new Exception("Error: Classifier has already been built. Unable to change classifiers within ensemble.");
		}
		this.classifiersToUse = new TreeSet<ClassifierVariants>();
		classifiersToUse.addAll(Arrays.asList(ClassifierVariants.values()));
	}
	
	public void turnSDMClassifiersOn() throws Exception {
		if (classifierBuilt) {
			throw new Exception("Error: Classifier has already been built. Unable to change classifiers within ensemble.");
		}
		this.classifiersToUse = new TreeSet<ClassifierVariants>();
		classifiersToUse.add(ClassifierVariants.Euclidean_1NN);
		classifiersToUse.add(ClassifierVariants.DTW_R1_1NN);
		classifiersToUse.add(ClassifierVariants.DTW_Rn_1NN);
		classifiersToUse.add(ClassifierVariants.WDTW_1NN);
		classifiersToUse.add(ClassifierVariants.DDTW_R1_1NN);
		classifiersToUse.add(ClassifierVariants.DDTW_Rn_1NN);
		classifiersToUse.add(ClassifierVariants.WDDTW_1NN);
		classifiersToUse.add(ClassifierVariants.LCSS_1NN);
		classifiersToUse.add(ClassifierVariants.TWE_1NN);
	}
	
	//add by jc
	public void turnSomeClassifiersOn() throws Exception {
		if (classifierBuilt) {
			throw new Exception("Error: Classifier has already been built. Unable to change classifiers within ensemble.");
		}
		this.classifiersToUse = new TreeSet<ClassifierVariants>();
		classifiersToUse.add(ClassifierVariants.Euclidean_1NN);
		classifiersToUse.add(ClassifierVariants.DTW_R1_1NN);
		classifiersToUse.add(ClassifierVariants.WDTW_1NN);
		classifiersToUse.add(ClassifierVariants.DDTW_R1_1NN);
		classifiersToUse.add(ClassifierVariants.WDDTW_1NN);
		classifiersToUse.add(ClassifierVariants.LCSS_1NN);
		classifiersToUse.add(ClassifierVariants.MSM_1NN);
		classifiersToUse.add(ClassifierVariants.ERP_1NN);
	}

	public boolean addClassifierToEnsemble(ClassifierVariants classifierToAdd) throws Exception {
		if (classifierBuilt) {
			throw new Exception("Error: Classifier has already been built. Unable to change classifiers within ensemble.");
		}
		if (this.classifiersToUse.contains(classifierToAdd)) {
			return false;
		} else {
			classifiersToUse.add(classifierToAdd);
			return true;
		}
	}

	public boolean removeClassifierFromEnsemble(ClassifierVariants classifierToRemove) throws Exception {
		if (classifierBuilt) {
			throw new Exception("Error: Classifier has already been built. Unable to change classifiers within ensemble.");
		}
		if (this.classifiersToUse.contains(classifierToRemove)) {
			this.classifiersToUse.remove(classifierToRemove);
			return true;
		} else {
			return false;
		}
	}

	public boolean removeAllClassifiersFromEnsemble() throws Exception {
		if (classifierBuilt) {
			throw new Exception("Error: Classifier has already been built. Unable to change classifiers within ensemble.");
		}
		this.classifiersToUse = new TreeSet<ClassifierVariants>();

		return true;
	}

	public void setEnsembleType(EnsembleType ensembleType) {
		this.ensembleType = ensembleType;
	}

	public void turnOnFileWriting(String outputDirLocation, String datasetName) {
		this.fileWriting = true;
		this.outpurDirLocation = outputDirLocation;
		this.datasetName = datasetName;
	}

	public void makeVerbose() {
		this.verbose = true;
	}

	@Override
	public void buildClassifier(Instances train) throws Exception {
		// if applicable, check that the file locations are valid before
		// carrying out cv

		File cvDir = null;

		File classifierOutputDir;
		StringBuilder st;
		StringBuilder bestParamsSt = new StringBuilder();
		FileWriter cvOut;
		int correct;

		this.trainActualClassVals = new double[train.numInstances()];
		for (int ins = 0; ins < trainActualClassVals.length; ins++) {
			trainActualClassVals[ins] = train.instance(ins).classValue();
		}

		if (fileWriting) {
			cvDir = new File(this.outpurDirLocation + "/bestCvOut");
			boolean valid = cvDir.mkdirs();

			if (!valid && !cvDir.exists()) {
				throw new Exception("The output dir at: " + outpurDirLocation + " could not be created.");
			} else if (!valid) {
				System.out.println("Warning: cvDir at " + this.outpurDirLocation + " already exists. Any conflicting results under this location will be overwritten.");
			}
		}

		this.finalClassifiers = new ClassifierVariants[this.classifiersToUse.size()];
		int c = 0;
		for (ClassifierVariants classifier : this.classifiersToUse) {
			this.finalClassifiers[c++] = classifier;
		}
		// carry out the cross validation

		this.cvAccs = new double[this.finalClassifiers.length];
		this.cvPreds = new double[this.finalClassifiers.length][train.numInstances()];
		this.bestParams = new double[this.finalClassifiers.length][];

		for (int i = 0; i < finalClassifiers.length; i++) {
			ClassifierVariants classifierType = this.finalClassifiers[i];

			crossValidateClassifierType(classifierType, train, i);
			if (fileWriting) {
				classifierOutputDir = new File(this.outpurDirLocation + "/bestCvOut/" + classifierType);
				classifierOutputDir.mkdirs();
				st = new StringBuilder();
				correct = 0;
				for (int j = 0; j < this.cvPreds[i].length; j++) {
					st.append(this.cvPreds[i][j]).append(",").append(trainActualClassVals[j]).append("\n");
					if (this.cvPreds[i][j] == trainActualClassVals[j]) {
						correct++;
					}
				}
				cvOut = new FileWriter(this.outpurDirLocation + "/bestCvOut/" + classifierType + "/" + "cvBest_" + classifierType + "_" + datasetName + ".txt");
				cvOut.append(correct + "/" + cvPreds[i].length + "\n");
				cvOut.append(st);
				cvOut.close();
				bestParamsSt.append(classifierType).append(",");
				for (int j = 0; bestParams[i] != null && j < bestParams[i].length; j++) {
					bestParamsSt.append(bestParams[i][j]).append(",");
				}
				bestParamsSt.append("\n");
			}
		}

		if (fileWriting) {
			File paramsOutputDir = new File(this.outpurDirLocation + "/bestParams/");
			paramsOutputDir.mkdirs();
			FileWriter bestParamsOut = new FileWriter(this.outpurDirLocation + "/bestParams/bestParams_" + this.datasetName + ".txt");
			bestParamsOut.append(bestParamsSt);
			bestParamsOut.close();
		}

		this.fullTrainingData = train;

		if (this.ensembleType == EnsembleType.Signif) {
			mcNemarsInclusion = this.getMcNemarsInclusion();
		}
		this.classifierBuilt = true;
	}

	private void crossValidateClassifierType(ClassifierVariants classifierType, Instances inputTrainingData, int classifierNum) throws Exception {
		Instances train;
		// prepare for derivative classifiers
		if (classifierType.equals(ClassifierVariants.DDTW_R1_1NN) || classifierType.equals(ClassifierVariants.DDTW_Rn_1NN) || classifierType.equals(ClassifierVariants.WDDTW_1NN)) {
			DerivativeFilter d = new DerivativeFilter();
			train = d.process(inputTrainingData);
		} else {
			train = inputTrainingData;
		}

		long startTime = -1;
		if (verbose) {
			System.out.print("Starting CV on " + classifierType + "...");
			startTime = System.nanoTime();
		}

		this.cvAccs[classifierNum] = -1;
		double[] params;
		CvOutput result;

		switch (classifierType) {
		// single-run classifiers (i.e. no params to tune, cv only needed for
		// weighting in ensemble later)
		case Euclidean_1NN:
		case DTW_R1_1NN:
		case DDTW_R1_1NN:
			result = crossValidate(train, classifierType, null);
			this.cvAccs[classifierNum] = result.getAccuracy();
			this.cvPreds[classifierNum] = result.getPredictions();
			this.bestParams[classifierNum] = null;

			break;
		// window-based/weight-based classifiers (i.e. 0:0.01:1)
		case DTW_Rn_1NN:
		case WDTW_1NN:
		case DDTW_Rn_1NN:
		case WDDTW_1NN:
			params = new double[1];
			// values range from 0 to 1 in increments of 0.01; use ints to avoid
			// double imprecision when incrementing
			for (int w = 0; w <= 100; w++) {
				params[0] = (double) w / 100;
				result = crossValidate(train, classifierType, params);
				if (result.getAccuracy() > this.cvAccs[classifierNum]) { // favours
																			// smaller
																			// window
																			// sizes
					this.cvAccs[classifierNum] = result.getAccuracy();
					this.cvPreds[classifierNum] = result.getPredictions();
					this.bestParams[classifierNum] = Arrays.copyOf(params, params.length);
				}
			}
			break;
		case LCSS_1NN:
			// params depend on data - get these from class
			double stdTrain = LCSSDistance.stdv_p(train);
			double stdFloor = stdTrain * 0.2;
			double[] epsilons = LCSSDistance.getInclusive10(stdFloor, stdTrain);
			int[] deltas = LCSSDistance.getInclusive10(0, (train.numAttributes() - 1) / 4);

			params = new double[2];
			for (int d = 0; d < deltas.length; d++) {
				params[0] = deltas[d];
				for (int e = 0; e < epsilons.length; e++) {
					params[1] = epsilons[e];
					result = crossValidate(train, classifierType, params);
					if (result.getAccuracy() > this.cvAccs[classifierNum]) {
						this.cvAccs[classifierNum] = result.getAccuracy();
						this.cvPreds[classifierNum] = result.getPredictions();
						this.bestParams[classifierNum] = Arrays.copyOf(params, params.length);
					} else if (result.getAccuracy() == this.cvAccs[classifierNum] && params[0] < epsilons[e] && params[1] < deltas[d]) {
						this.cvAccs[classifierNum] = result.getAccuracy();
						this.cvPreds[classifierNum] = result.getPredictions();
						this.bestParams[classifierNum] = Arrays.copyOf(params, params.length);
					}
				}
			}
			break;
		case MSM_1NN:
			params = new double[1];
			// values have a variable range. Specified in a static array at the
			// start of the class called msmParams
			for (int p = 0; p < msmParms.length; p++) {
				params[0] = msmParms[p];
				result = crossValidate(train, classifierType, params);
				if (result.getAccuracy() > this.cvAccs[classifierNum]) { // favours
																			// smaller
																			// params
					this.cvAccs[classifierNum] = result.getAccuracy();
					this.cvPreds[classifierNum] = result.getPredictions();
					this.bestParams[classifierNum] = Arrays.copyOf(params, params.length);
				}
			}
			break;
		case TWE_1NN:
			// values have variable ranges, so are specified in static arrays at
			// the top of the class as twe_nuParams and twe_lambdaParams
			params = new double[2];
			for (int n = 0; n < twe_nuParams.length; n++) {
				params[0] = twe_nuParams[n];
				for (int la = 0; la < twe_lamdaParams.length; la++) {
					params[1] = twe_lamdaParams[la];
					result = crossValidate(train, classifierType, params);
					if (result.getAccuracy() > this.cvAccs[classifierNum]) {
						this.cvAccs[classifierNum] = result.getAccuracy();
						this.cvPreds[classifierNum] = result.getPredictions();
						this.bestParams[classifierNum] = Arrays.copyOf(params, params.length);
					} else if (result.getAccuracy() == this.cvAccs[classifierNum] && params[0] < twe_nuParams[n] && params[1] < twe_lamdaParams[la]) {
						this.cvAccs[classifierNum] = result.getAccuracy();
						this.cvPreds[classifierNum] = result.getPredictions();
						this.bestParams[classifierNum] = Arrays.copyOf(params, params.length);
					}
				}
			}
			break;
		case ERP_1NN:
			// values of g depend on the data, so get the standard deviation and
			// then work them out. Note: window of 0-25% used as per Keogh et
			// al.'s paper. Sampled to
			// produce 100 different paramater combinations in total
			double[] windowSizes = ERPDistance.getInclusive10(0, 0.25);
			double stdv = ERPDistance.stdv_p(train);
			double[] gValues = ERPDistance.getInclusive10(0.2 * stdv, stdv);
			params = new double[2];
			// g bandsize
			for (int w = 0; w < windowSizes.length; w++) {
				params[1] = windowSizes[w];
				for (int g = 0; g < gValues.length; g++) {
					params[0] = gValues[g];
					result = crossValidate(train, classifierType, params);
					if (result.getAccuracy() > this.cvAccs[classifierNum]) {
						this.cvAccs[classifierNum] = result.getAccuracy();
						this.cvPreds[classifierNum] = result.getPredictions();
						this.bestParams[classifierNum] = Arrays.copyOf(params, params.length);
					} else if (result.getAccuracy() == this.cvAccs[classifierNum] && params[0] < gValues[g] && params[1] < windowSizes[w]) {
						this.cvAccs[classifierNum] = result.getAccuracy();
						this.cvPreds[classifierNum] = result.getPredictions();
						this.bestParams[classifierNum] = Arrays.copyOf(params, params.length);
					}
				}
			}
			break;
		default:
			throw new Exception("The classifier type \"" + classifierType + "\" is not defined within the ensemble. Please update the code.");
		}
		if (verbose) {
			System.out.println("Done! (in " + ((System.nanoTime() - startTime) / 1000000000) + " seconds)");
		}

	}

	private static class IndividualClassificationOutput {
		private int id;
		private double prediction;

		public IndividualClassificationOutput(int id, double prediction) {
			this.id = id;
			this.prediction = prediction;
		}

		public int getId() {
			return id;
		}

		public double getPrediction() {
			return prediction;
		}

	}

	private static class SingleCVCaller implements Callable<IndividualClassificationOutput> {
		private Instances train;
		private ClassifierVariants classifierType;
		private double[] params;
		private int i;

		public SingleCVCaller(Instances train, ClassifierVariants classifierType, double[] params, int i) {
			this.train = train;
			this.classifierType = classifierType;
			this.params = params;
			this.i = i;
		}

		@Override
		public IndividualClassificationOutput call() throws Exception {
			Instance testInstance;
			Instances trainLoocv;
			kNN knn;

			testInstance = train.instance(i);
			trainLoocv = new Instances(train, train.numInstances() - 1);

			// add all instances to trainLoocv EXCEPT instance[i]
			for (int j = 0; j < train.numInstances(); j++) {
				if (j != i) {
					trainLoocv.add(train.instance(j));
				}
			}

			// build classifier and classify
			knn = getInternalClassifier(classifierType, params, trainLoocv);
			return new IndividualClassificationOutput(i, knn.classifyInstance(testInstance));
		}
	}

	private static class SingleTrainTestCaller implements Callable<IndividualClassificationOutput> {
		private int i;
		private Instance testInstance;
		private kNN classifier;

		public SingleTrainTestCaller(int i, Instance testInstance, kNN classifier) {
			this.i = i;
			this.testInstance = testInstance;
			this.classifier = classifier;
		}

		@Override
		public IndividualClassificationOutput call() throws Exception {
			return new IndividualClassificationOutput(i, classifier.classifyInstance(testInstance));
		}
	}

	private static CvOutput crossValidate(Instances train, ClassifierVariants classifierType, double[] params) throws Exception {

		double[] predictions = new double[train.numInstances()];

		int correct = 0;
		int total = 0;

		ExecutorService service = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
		ArrayList<Future<IndividualClassificationOutput>> futures = new ArrayList<Future<IndividualClassificationOutput>>();

		for (int i = 0; i < train.numInstances(); i++) {
			futures.add(service.submit(new SingleCVCaller(train, classifierType, params, i)));
		}
		service.shutdown();

		IndividualClassificationOutput result;
		for (int i = 0; i < futures.size(); i++) {
			result = futures.get(i).get();
			predictions[result.id] = result.prediction;
			if (predictions[result.id] == train.instance(result.id).classValue()) {
				correct++;
			}
			total++;
		}

		CvOutput output = new CvOutput(100.0 / total * correct, predictions);
		return output;
	}

	protected static kNN getInternalClassifier(ClassifierVariants classifierType, double[] params, Instances instances) throws Exception {

		EuclideanDistance distanceMeasure = null;
		kNN knn;
		switch (classifierType) {
		case Euclidean_1NN:
			distanceMeasure = new EuclideanDistance();
			distanceMeasure.setDontNormalize(true);
			break;
		case DTW_R1_1NN:
		case DDTW_R1_1NN:
			distanceMeasure = new BasicDTW();
			break;
		case DTW_Rn_1NN:
		case DDTW_Rn_1NN:
			distanceMeasure = new SakoeChibaDTW(params[0]);
			break;
		case WDTW_1NN:
		case WDDTW_1NN:
			distanceMeasure = new WeightedDTW(params[0]);
			break;
		case LCSS_1NN:
			distanceMeasure = new LCSSDistance((int) params[0], params[1]);
			break;
		case MSM_1NN:
			distanceMeasure = new MSMDistance(params[0]);
			break;
		case TWE_1NN:
			distanceMeasure = new TWEDistance(params[0], params[1]);
			break;
		case ERP_1NN:
			distanceMeasure = new ERPDistance(params[0], params[1]);
			break;
		default:
			throw new Exception("Error: " + classifierType + " is not a supported classifier type. Please update code to use this in the ensemble");
		}

		knn = new kNN();
		knn.setDistanceFunction(distanceMeasure);
		knn.buildClassifier(instances);
		return knn;
	}

	protected static double[] getParamsFromParamId(ClassifierVariants classifierType, int paramId, Instances trainingData) throws Exception {
		double[] out;
		switch (classifierType) {
		case Euclidean_1NN:
		case DTW_R1_1NN:
		case DDTW_R1_1NN:
			return null;
		case DTW_Rn_1NN:
		case DDTW_Rn_1NN:
		case WDTW_1NN:
		case WDDTW_1NN:
			out = new double[1];
			out[0] = ((double) paramId) / 100;
			return out;
		case LCSS_1NN:
			double stdTrain = LCSSDistance.stdv_p(trainingData);
			double stdFloor = stdTrain * 0.2;
			double[] epsilons = LCSSDistance.getInclusive10(stdFloor, stdTrain);
			int[] deltas = LCSSDistance.getInclusive10(0, (trainingData.numAttributes() - 1) / 4);
			return new double[] { deltas[paramId / 10], epsilons[paramId % 10] };
		case MSM_1NN:
			return new double[] { msmParms[paramId] };
		case TWE_1NN:
			return new double[] { twe_nuParams[paramId / 10], twe_lamdaParams[paramId % 10] };
		case ERP_1NN:
			double[] windowSizes = ERPDistance.getInclusive10(0, 0.25);
			double stdv = ERPDistance.stdv_p(trainingData);
			double[] gValues = ERPDistance.getInclusive10(0.2 * stdv, stdv);

			return new double[] { gValues[paramId / 10], windowSizes[paramId % 10] };
		default:
			throw new Exception("Error: " + classifierType + " is not a supported classifier type. Please update code to use this in the ensemble");
		}
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		if (!classifierBuilt) {
			throw new Exception("Error: Classifier has not been built! Classifier must be built before carrying out classification. See buildClassifier(Instances train).");
		}

		// special case if classifier is originally built for a non-signif
		// ensemble, but then changed after building (this is valid, as the cv
		// remains the same but allows for
		// classification using any of the ensembling strategies). This is
		// necessarry as the getMcNemarsInclusion() call is originally in
		// buildClassifier(Instances train) for
		// efficiency, as it depends on the original ensemble stratergy when the
		// build function is executed (no point working it out if it's not being
		// used!).
		if (this.ensembleType == EnsembleType.Signif && this.mcNemarsInclusion == null) {
			this.mcNemarsInclusion = getMcNemarsInclusion();
		}

		int numProcessors = Runtime.getRuntime().availableProcessors();
		// int numThreads = (numProcessors > this.finalClassifiers.length) ?
		// this.finalClassifiers.length:numProcessors;
		ExecutorService service = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
		ArrayList<Future<IndividualClassificationOutput>> futures = new ArrayList<Future<IndividualClassificationOutput>>();
		predictions = new double[this.finalClassifiers.length];
		ClassifierVariants classifier;
		for (int i = 0; i < predictions.length; i++) {
			classifier = this.finalClassifiers[i];
			if (this.ensembleType != EnsembleType.Signif || this.mcNemarsInclusion[i] == true) {
				kNN knn = getInternalClassifier(classifier, this.bestParams[i], this.fullTrainingData);
				futures.add(service.submit(new SingleTrainTestCaller(i, instance, knn)));
			} else {
				predictions[i] = -1;
			}
		}
		service.shutdown();

		IndividualClassificationOutput result;
		for (int i = 0; i < futures.size(); i++) {
			result = futures.get(i).get();
			predictions[result.id] = result.prediction;
		}

		switch (this.ensembleType) {
		case Best:
			return this.classifyInstances_best(predictions);
		case Equal:
			return this.classifyInstances_equal(predictions);
		case Prop:

			return this.classifyInstances_prop(predictions);
		case Signif:
			return this.classifyInstances_prop(predictions);
		default:
			throw new Exception("Error: Unexpected ensemble type");
		}
	}

	public double[] getPredictions() {
		return predictions;
	}

	private double classifyInstances_best(double[] predictions) {

		ArrayList<Integer> bestClassifierIds = new ArrayList<Integer>();
		double bsfAcc = -1;
		for (int i = 0; i < this.cvAccs.length; i++) {
			if (this.cvAccs[i] > bsfAcc) {
				bestClassifierIds = new ArrayList<Integer>();
				bestClassifierIds.add(i);
				bsfAcc = this.cvAccs[i];
			} else if (this.cvAccs[i] == bsfAcc) {
				bestClassifierIds.add(i);
			}
		}
		if (bestClassifierIds.size() > 1) {
			Random r = new Random();
			return predictions[r.nextInt(bestClassifierIds.size())];
		} else {
			return predictions[bestClassifierIds.get(0)];
		}
	}

	private double classifyInstances_equal(double[] predictions) {

		TreeMap<Double, Integer> classValsAndVotes = new TreeMap<Double, Integer>();
		for (int c = 0; c < predictions.length; c++) {
			double thisVote = predictions[c];
			if (classValsAndVotes.containsKey(thisVote)) {
				int currentCount = classValsAndVotes.get(thisVote);
				currentCount++;
				classValsAndVotes.put(thisVote, currentCount);
			} else {
				classValsAndVotes.put(thisVote, 1);
			}
		}

		ArrayList<Double> majorityClasses = new ArrayList<Double>();
		int bsfCount = -1;
		int thisCount;
		for (Double classVal : classValsAndVotes.keySet()) {
			thisCount = classValsAndVotes.get(classVal);
			if (thisCount > bsfCount) {
				bsfCount = thisCount;
				majorityClasses = new ArrayList<Double>();
				majorityClasses.add(classVal);
			} else if (thisCount == bsfCount) {
				majorityClasses.add(classVal);
			}
		}

		if (majorityClasses.size() == 1) {
			return majorityClasses.get(0);
		} else {
			Random r = new Random();
			return majorityClasses.get(r.nextInt(majorityClasses.size()));
		}
	}

	private double classifyInstances_prop(double[] predictions) {

		TreeMap<Double, Double> classValsAndVotes = new TreeMap<Double, Double>();

		double thisVote;
		double currentWeight;

		double bsfWeight = 0;
		ArrayList<Double> majorityClasses = new ArrayList<Double>();

		for (int c = 0; c < predictions.length; c++) {
			thisVote = predictions[c];

			if (this.ensembleType == EnsembleType.Prop || this.mcNemarsInclusion[c] == true) {
				if (classValsAndVotes.containsKey(thisVote)) {
					currentWeight = classValsAndVotes.get(thisVote);
					currentWeight += cvAccs[c];
					classValsAndVotes.put(thisVote, currentWeight);
				} else {
					currentWeight = cvAccs[c];
					classValsAndVotes.put(thisVote, currentWeight);
				}

				if (currentWeight > bsfWeight) {
					majorityClasses = new ArrayList<Double>();
					majorityClasses.add(thisVote);
					bsfWeight = currentWeight;
				} else if (currentWeight == bsfWeight) {
					majorityClasses.add(thisVote);
				}
			}
		}

		if (majorityClasses.size() == 1) {
			return majorityClasses.get(0);
		} else {
			Random r = new Random();
			return majorityClasses.get(r.nextInt(majorityClasses.size()));
		}
	}

	public static void buildAndWriteCvAndTrainTestFiles_SDM(String outDir, String datasetName, Instances train, Instances test) throws Exception {

		ElasticEnsemble elastic = new ElasticEnsemble();
		elastic.setEnsembleType(EnsembleType.Best); // Doesn't matter
		elastic.turnSDMClassifiersOn();
		elastic.turnOnFileWriting(outDir, datasetName);
		elastic.makeVerbose();
		elastic.buildClassifier(train);

		kNN knn;
		File outputDir;
		FileWriter out;
		StringBuilder st;
		int correct, total;
		double decision, classValue;
		ClassifierVariants classifier;

		for (int c = 0; c < elastic.finalClassifiers.length; c++) {
			classifier = elastic.finalClassifiers[c];
			knn = getInternalClassifier(classifier, elastic.bestParams[c], train);

			correct = 0;
			total = 0;
			st = new StringBuilder();
			for (int i = 0; i < test.numInstances(); i++) {
				decision = knn.classifyInstance(test.instance(i));
				classValue = test.instance(i).classValue();
				if (decision == classValue) {
					correct++;
				}
				total++;
				st.append(decision).append(",").append(classValue).append("\n");
			}
			outputDir = new File(outDir + "/trainTest/" + classifier);
			outputDir.mkdirs();
			out = new FileWriter(outDir + "/trainTest/" + classifier + "/trainTest_" + classifier + "_" + datasetName + ".txt");
			out.append(correct + "/" + total + "\n");
			out.append(st);
			out.close();

		}
	}

	public static void demonstrateEnsembles_SettingsFromSDM(Instances train, Instances test, String outputDir, String datasetName) throws Exception {

		// 1. Initialise classifier in the usual Weka form
		ElasticEnsemble elastic = new ElasticEnsemble();

		// 2. Set the internal distance measure 1NN classifiers to use. By
		// default, the ensemble classifier won't use any (and will just throw
		// an error).
		// These can be specified individually, i.e.
		// this.addClassifierToEnsemble(ClassifierVariants.Euclidean1NN), or
		// there are two special cases:
		// - this.turnAllClassifiersOn() // Uses all possible classifiers that
		// have been written for the ensemble (DAMI version including TWED and
		// ERO)
		// - this.turnSDMClassifiersOn() // Uses the classifiers that were
		// included in the SDM paper
		// For the purposes of this demonstration, we will turn on all
		// classifiers:
		// elastic.turnSDMClassifiersOn();
		elastic.turnAllClassifiersOn();

		// 3. By default, the ensmeble works like a typical Weka classifier -
		// i.e. you build it, run it, and then it is removed from memory.
		// However, it can also be set to write the cv results to file (i.e. for
		// faster build times when repeating experiments (NOT IMPLEMENTED HERE),
		// or for information).
		// Files are written to the dir specified by the String outputDir (and
		// creates it/parent dirs if necessary), and files arenames using the
		// name
		// specified in datasetName. It is important to keep this consistent
		// with the data for easily reusing cv results
		// (i.e. if training with ItalyPowerDemand_TRAIN.arff, use
		// ItalyPowerDemand as the datasetName so the arff can be found
		// dynamically late on)
		// ***IMPORTANT: Will overwrite existing files as necessary. It will
		// continue to use existing dirs (i.e. if a different dataset has been
		// processed,
		// those files will remain unchanged
		// elastic.turnOnFileWriting(outputDir, datasetName);

		// 4. Training can be slow with large datasets and many classifiers in
		// the ensemble. For peace of mind, a method is included to promt the
		// classifier
		// to print messages to the system output during training to state which
		// distance measure is currently being processed (and the time taken to
		// complete
		// once it has been done).
		elastic.makeVerbose();

		// 5. Build the classifier on the specified training data
		elastic.buildClassifier(train);

		int correct, total;
		double prediction, classValue;
		DecimalFormat df = new DecimalFormat("###.###");

		System.out.println();
		System.out.println("-----------------------------------------");
		System.out.println("TRAIN/TEST CLASSIFICATION");
		System.out.println("-----------------------------------------");

		// To save time and create a fair comparison, we build once and then
		// classify separately for each ensemble strategy. This is valid, as
		// ensemble type is
		// completely independent from the CV in the training stage of the
		// classifier, so would be the same if we carried it our separately for
		// each ensemble
		EnsembleType[] types = { EnsembleType.Best, EnsembleType.Equal, EnsembleType.Prop, EnsembleType.Signif };
		for (int t = 0; t < types.length; t++) {
			elastic.setEnsembleType(types[t]);
			correct = 0;
			total = 0;

			for (int i = 0; i < test.numInstances(); i++) {
				prediction = elastic.classifyInstance(test.instance(i));
				classValue = test.instance(i).classValue();
				if (prediction == classValue) {
					correct++;
				}
				total++;
			}
			System.out.println(elastic.ensembleType + ": " + correct + "/" + total + " (" + df.format(100.0 / total * correct) + "%)");
		}
	}

	public boolean[] getMcNemarsInclusion() {
		// find the best classifier according to cvAccuracies - random selection
		// of best where ties are equal
		ArrayList<Integer> bestClassifiersIds = new ArrayList<Integer>();
		double bsfAccuracy = -1;

		for (int c = 0; c < cvAccs.length; c++) {
			if (cvAccs[c] > bsfAccuracy) {
				bestClassifiersIds = new ArrayList<Integer>();
				bestClassifiersIds.add(c);
			} else if (cvAccs[c] == bsfAccuracy) {
				bestClassifiersIds.add(c);
			}
		}

		int bestClassifierId = -1;
		if (bestClassifiersIds.size() == 1) {
			bestClassifierId = bestClassifiersIds.get(0);
		} else {
			Random r = new Random();
			bestClassifierId = bestClassifiersIds.get(r.nextInt(bestClassifiersIds.size()));
		}

		// go through each classifier and calculate McNemars. For each
		// classifier, add either a 1 or 0 to the output to reflect whether the
		// classifier should be used in the array. i.e. if a classifier is
		// significantly different to the best (i.e. it must be worse), output 0
		// for
		// that classifier. Else, output 1 to show that it is not significantly
		// worse, and sjouldbe included.

		boolean[] output = new boolean[this.finalClassifiers.length];

		for (int classifierB = 0; classifierB < this.finalClassifiers.length; classifierB++) {

			if (classifierB == bestClassifierId) {
				output[classifierB] = true; // looking at itself, and obviously
											// we want the best classifier
											// included!
				continue;
			}

			// can include speedup where a==b, keep it simple for now until it's
			// working
			int wrongByBoth = 0; // top-left
			int rightByAWrongByB = 0; // bottom-left
			int wrongByARightByB = 0; // top-right
			int rightByBoth = 0; // bottom-right

			double actualClass, thisPred, bPred;

			for (int i = 0; i < this.trainActualClassVals.length; i++) {
				actualClass = trainActualClassVals[i];
				thisPred = cvPreds[bestClassifierId][i];
				bPred = cvPreds[classifierB][i];

				if (thisPred != actualClass && bPred != actualClass) {
					wrongByBoth++;
				} else if (thisPred == actualClass && bPred != actualClass) {
					rightByAWrongByB++;
				} else if (thisPred != actualClass && bPred == actualClass) {
					wrongByARightByB++;
				} else if (thisPred == actualClass && bPred == actualClass) {
					rightByBoth++;
				}
			}

			if (rightByAWrongByB + wrongByARightByB == 0) {
				output[classifierB] = true; // classifier is equivilent to the
											// best, so we should include it to
											// effectively add weight to best's
											// votes
			} else {
				double chiPart = (Math.abs(wrongByARightByB - rightByAWrongByB) - 1);
				double chi = (chiPart * chiPart) / (wrongByARightByB + rightByAWrongByB);

				if (chi >= 3.841459) { // Alpha = 0.05
					// if(chi >= 6.634897){ // Alpha = 0.01
					output[classifierB] = false; // signif. different, so don't
													// include
				} else {
					output[classifierB] = true; // no signif different (i.e. not
												// signif worse), so include
				}
			}
		}
		return output;
	}

	protected static class CvOutput {
		private double accuracy;
		private double[] predictions;
		private double[] params;

		public CvOutput(double accuracy, double[] predictions) {
			this.accuracy = accuracy;
			this.predictions = predictions;
		}

		public CvOutput(double accuracy, double[] predictions, double[] params) {
			this.accuracy = accuracy;
			this.predictions = predictions;
			this.params = Arrays.copyOf(params, params.length);
		}

		public double getAccuracy() {
			return this.accuracy;
		}

		public double[] getPredictions() {
			return this.predictions;
		}

		public double[] getParams() {
			return params;
		}

	}

	@Override
	public String toString() {
		return this.classifiersToUse.toString();
	}

	public static void mainExample(String[] args) {

		try {
			// Example use of the classifier for dataset ItalyPowerDemand.
			// Please see demonstrateEnsembles...
			// Dataset:
			String datasetName = "ItalyPowerDemand";

			// Data dir:
			String dataDir = "../../TSC Problems";

			if (!new File(dataDir).exists()) {
				throw new Exception("Error: Specified data directory does not exist: " + dataDir);
			}

			Instances train = ClassifierTools.loadData(dataDir + "/" + datasetName + "/" + datasetName + "_TRAIN.arff");
			Instances test = ClassifierTools.loadData(dataDir + "/" + datasetName + "/" + datasetName + "_TEST.arff");

			// see method for annotations
			demonstrateEnsembles_SettingsFromSDM(train, test, "demonstration", datasetName);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void debugEE(Instances train, Instances test, OutFile of) throws Exception {
		// Build all the classifiers
		// Elastic ensemble

		ElasticEnsemble ee = new ElasticEnsemble();
		ee.turnAllClassifiersOn();
		ee.setEnsembleType(ElasticEnsemble.EnsembleType.Prop);
		ee.buildClassifier(train);
		System.out.println("BUILT EE");

		// We need the training CV for all the ensembles.
		double[] elasticCVAccs = ee.getCVAccs();
		if (elasticCVAccs == null)
			System.out.println("CV ACCS IS NULL");
		System.out.println("Train CV ACC (" + elasticCVAccs.length + ")");
		of.writeLine("Train CV ACC (" + elasticCVAccs.length + ")");
		of.writeString(",");

		for (double d : elasticCVAccs) {
			of.writeString((d / 100.0) + ",");
			System.out.print((d / 100.0) + ",");
		}
		of.writeString("\n");
		System.out.print("\n");
		for (int i = 0; i < test.numInstances(); i++) {
			// Get predictions for all
			ee.classifyInstance(test.instance(i));
			System.out.println("CLASSIFIED EE instance " + i);

			double[] eePreds = ee.getPredictions();
			for (double d : eePreds)
				of.writeString(d + ",");
			// Write all preds to file
			of.writeString("\n");
			if (i % 10 == 0)
				System.out.println("Finished case " + (i + 1) + " of " + test.numInstances());

		}

	}

}
