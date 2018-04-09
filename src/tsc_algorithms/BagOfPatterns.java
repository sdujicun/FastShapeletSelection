package tsc_algorithms;

import development.DataSets;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.kNN;
import weka.core.Capabilities;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.timeseries.BagOfPatternsFilter;
import weka.filters.timeseries.SAX;

/**
 * Converts instances into Bag Of Patterns form, then gives to a 1NN
 * 
 * Params: wordLength, alphabetSize, windowLength
 * 
 * @author James
 */
public class BagOfPatterns implements Classifier {

	public Instances matrix;
	public kNN knn;

	private BagOfPatternsFilter bop;
	private int PAA_intervalsPerWindow;
	private int SAX_alphabetSize;
	private int windowSize;

	private FastVector alphabet;

	private final boolean useParamSearch; // does user want parameter search to
											// be performed

	/**
	 * No params given, do parameter search
	 */
	public BagOfPatterns() {
		this.PAA_intervalsPerWindow = -1;
		this.SAX_alphabetSize = -1;
		this.windowSize = -1;

		
		knn = new kNN(); // defaults to 1NN, Euclidean distance
		useParamSearch = true;
	}

	/**
	 * Params given, use those only
	 */
	public BagOfPatterns(int PAA_intervalsPerWindow, int SAX_alphabetSize, int windowSize) {
		this.PAA_intervalsPerWindow = PAA_intervalsPerWindow;
		this.SAX_alphabetSize = SAX_alphabetSize;
		this.windowSize = windowSize;

		bop = new BagOfPatternsFilter(PAA_intervalsPerWindow, SAX_alphabetSize, windowSize);
		knn = new kNN(); // default to 1NN, Euclidean distance
		alphabet = SAX.getAlphabet(SAX_alphabetSize);

		useParamSearch = false;
	}

	public int getPAA_intervalsPerWindow() {
		return PAA_intervalsPerWindow;
	}

	public int getSAX_alphabetSize() {
		return SAX_alphabetSize;
	}

	public int getWindowSize() {
		return windowSize;
	}

	/**
	 * @return { numIntervals(word length), alphabetSize, slidingWindowSize }
	 */
	public int[] getParameters() {
		return new int[] { PAA_intervalsPerWindow, SAX_alphabetSize, windowSize };
	}

	/**
	 * Performs cross validation on given data for varying parameter values,
	 * returns parameter set which yielded greatest accuracy
	 * 
	 * @param data
	 *            Data to perform cross validation testing on
	 * @return { numIntervals, alphabetSize, slidingWindowSize }
	 */
	public static int[] parameterSearch(Instances data) throws Exception {
		double bestAcc = 0.0;
		int bestAlpha = 0, bestWord = 0, bestWindowSize = 0;
		int numTests = 5;

		// BoP paper window search range suggestion
		int minWinSize = (int) ((data.numAttributes() - 1) * (5.0 / 100.0));
		if(minWinSize<=8)
			minWinSize=8;
		int maxWinSize = (int) ((data.numAttributes() - 1) * (50.0 / 100.0));
		if(maxWinSize<=8)
			maxWinSize=8;
//		int minWinSize=4;
//		int maxWinSize=data.numAttributes() - 1;
		// int winInc = 1; //check every size in range
		int winInc = (int) ((maxWinSize - minWinSize) /20.0); // check 10
														// values within
																// that range
		if (winInc < 1)
			winInc = 1;
		//int winInc = 1;	

		for (int alphaSize = 2; alphaSize <= 8; alphaSize++) {
			for (int winSize = minWinSize; winSize <= maxWinSize; winSize += winInc) {
				for (int wordSize = 4; wordSize <= winSize / 2&&wordSize <= 16; wordSize ++) { // lin
																					// BoP
																					// suggestion
//				for (int wordSize = 2; wordSize <= winSize / 2; wordSize ++ ) {
					BagOfPatterns bop = new BagOfPatterns(wordSize, alphaSize, winSize);
					double acc = bop.crossValidate(data); // leave-one-out
															// without rebuiding
															// every fold

					if (acc > bestAcc) {
						bestAcc = acc;
						bestAlpha = alphaSize;
						bestWord = wordSize;
						bestWindowSize = winSize;
					}
				}
			}
		}

		return new int[] { bestWord, bestAlpha, bestWindowSize };
	}

	/**
	 * Leave-one-out CV without re-doing identical transformation every fold
	 * 
	 * @return cv accuracy
	 */
	private double crossValidate(Instances data) throws Exception {
		buildClassifier(data);

		double correct = 0;
		for (int i = 0; i < data.numInstances(); ++i) {
			if (classifyInstance(i) == data.get(i).classValue()) {
				++correct;
			}
		}
		return correct / data.numInstances();
	}

	@Override
	public void buildClassifier(final Instances data) throws Exception {
		if (data.classIndex() != data.numAttributes() - 1)
			throw new Exception("LinBoP_BuildClassifier: Class attribute not set as last attribute in dataset");

		if (useParamSearch) {
			// find and set params
			int[] params = parameterSearch(data);

			this.PAA_intervalsPerWindow = params[0];
			this.SAX_alphabetSize = params[1];
			this.windowSize = params[2];

			bop = new BagOfPatternsFilter(PAA_intervalsPerWindow, SAX_alphabetSize, windowSize);
			alphabet = SAX.getAlphabet(SAX_alphabetSize);
		}

		// validate
		if (PAA_intervalsPerWindow < 0)
			throw new Exception("LinBoP_BuildClassifier: Invalid PAA word size: " + PAA_intervalsPerWindow);
		if (PAA_intervalsPerWindow > windowSize)
			throw new Exception("LinBoP_BuildClassifier: Invalid PAA word size, bigger than sliding window size: " + PAA_intervalsPerWindow + "," + windowSize);
		if (SAX_alphabetSize < 0 || SAX_alphabetSize > 10)
			throw new Exception("LinBoP_BuildClassifier: Invalid SAX alphabet size (valid=2-10): " + SAX_alphabetSize);
		if (windowSize < 0 || windowSize > data.numAttributes() - 1)
			throw new Exception("LinBoP_BuildClassifier: Invalid sliding window size: " + windowSize + " (series length " + (data.numAttributes() - 1) + ")");

		// real work
		matrix = bop.process(data); // transform
		knn.buildClassifier(matrix); // give to 1nn
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		// convert to BOP form
		double[] hist = bop.bagToArray(bop.buildBag(instance));

		// stuff into Instance
		Instances newInsts = new Instances(matrix, 1); // copy attribute data
		newInsts.add(new SparseInstance(1.0, hist));

		return knn.classifyInstance(newInsts.firstInstance());
	}

	/**
	 * Used as part of a leave-one-out crossvalidation, to skip having to
	 * rebuild the classifier every time (since n-1 histograms would be
	 * identical each time anyway), therefore this classifies the instance at
	 * the index passed while ignoring its own corresponding histogram
	 * 
	 * @param test
	 *            index of instance to classify
	 * @return classification
	 */
	public double classifyInstance(int test) {
		double bestDist = Double.MAX_VALUE;
		double nn = -1.0;

		Instance testInst = matrix.get(test);

		for (int i = 0; i < matrix.numInstances(); ++i) {
			if (i == test) // skip 'this' one, leave-one-out
				continue;

			double dist = knn.distance(testInst, matrix.get(i));

			if (dist < bestDist) {
				bestDist = dist;
				nn = matrix.get(i).classValue();
			}
		}

		return nn;
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		// convert to BOP form
		double[] hist = bop.bagToArray(bop.buildBag(instance));

		// stuff into Instance
		Instances newInsts = new Instances(matrix, 1); // copy attribute data
		newInsts.add(new SparseInstance(1.0, hist));

		return knn.distributionForInstance(newInsts.firstInstance());
	}

	@Override
	public Capabilities getCapabilities() {
		throw new UnsupportedOperationException("Not supported yet."); // To change body of generated/ methods, choose Tools| Templates.
	}

	public static void main(String[] args) {
		String[] problems = { "ChlorineConcentration", "Coffee", "ECGFiveDays", "GunPoint", "Lightning7", "MedicalImages", "MoteStrain", "Trace", "TwoLeadECG" };		
		System.out.println(problems.length);
		for (int i = 0; i < problems.length; i++) {
			System.out.print(problems[i] + "\t");
			basicTest(problems[i]);
		}
	}
	
	public static void basicTest(String problem) {
		try {
			String s=problem;
			Instances train = ClassifierTools.loadData(DataSets.problemPath + s + "\\" + s + "_TRAIN");
			Instances test = ClassifierTools.loadData(DataSets.problemPath + s + "\\" + s + "_TEST");
			BagOfPatterns bop = new BagOfPatterns();
			long start = System.nanoTime();
			bop.buildClassifier(train);
			double trainTime = (System.nanoTime() - start) / 1000000000.0; // seconds
			System.out.println("Training done (" + trainTime + "s)");
			start = System.nanoTime();
			double acc = ClassifierTools.accuracy(test, bop);
			double testTime = (System.nanoTime() - start) / 1000000000.0; // seconds
			System.out.println("Testing done (" + testTime + "s)");

			System.out.println("\nACC: " + acc);
		} catch (Exception e) {
			System.out.println(e);
			e.printStackTrace();
		}
	}

	public static void basicTest() {
		System.out.println("BOPBasicTest\n");
		try {
			//begin jc 2016-04-27
			String s="Car";
			Instances train = ClassifierTools.loadData(DataSets.problemPath + s + "\\" + s + "_TRAIN");
			Instances test = ClassifierTools.loadData(DataSets.problemPath + s + "\\" + s + "_TEST");
			//Instances train = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TRAIN.arff");
			//Instances test = ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TEST.arff");
			//end jc 2016-04-27
			
			// Instances train =
			// ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\BeetleFly\\BeetleFly_TRAIN.arff");
			// Instances test =
			// ClassifierTools.loadData("C:\\tempbakeoff\\TSC Problems\\BeetleFly\\BeetleFly_TEST.arff");

			System.out.println(train.relationName());

			BagOfPatterns bop = new BagOfPatterns();
			System.out.println("Training starting");
			long start = System.nanoTime();
			bop.buildClassifier(train);
			double trainTime = (System.nanoTime() - start) / 1000000000.0; // seconds
			System.out.println("Training done (" + trainTime + "s)");

			System.out.print("Params: ");
			for (int p : bop.getParameters())
				System.out.print(p + " ");
			System.out.println("");

			System.out.println("\nTesting starting");
			start = System.nanoTime();
			double acc = ClassifierTools.accuracy(test, bop);
			double testTime = (System.nanoTime() - start) / 1000000000.0; // seconds
			System.out.println("Testing done (" + testTime + "s)");

			System.out.println("\nACC: " + acc);
		} catch (Exception e) {
			System.out.println(e);
			e.printStackTrace();
		}
	}

	@Override
	public String toString() {
		return "BagOfPatterns";
	}
}
