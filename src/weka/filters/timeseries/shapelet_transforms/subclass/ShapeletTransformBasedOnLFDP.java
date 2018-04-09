/*
 * copyright: Anthony Bagnall
 * NOTE: As shapelet extraction can be time consuming, there is an option to output shapelets
 * to a text file (Default location is in the root dir of the project, file name "defaultShapeletOutput.txt").
 *
 * Default settings are TO NOT PRODUCE OUTPUT FILE - unless file name is changed, each successive filter will
 * overwrite the output (see "setLogOutputFile(String fileName)" to change file dir and name).
 *
 * To reconstruct a filter from this output, please see the method "createFilterFromFile(String fileName)".
 */
package weka.filters.timeseries.shapelet_transforms.subclass;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;

import development.DataSets;
import utilities.ClassifierTools;
import utilities.class_distributions.ClassDistribution;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.classifiers.trees.EnhancedRandomForest;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.core.elastic_distance_measures.DTW;
import weka.core.shapelet.*;
import weka.filters.SimpleBatchFilter;
import weka.filters.timeseries.shapelet_transforms.classValue.BinarisedClassValue;
import weka.filters.timeseries.shapelet_transforms.classValue.NormalClassValue;
import weka.filters.timeseries.shapelet_transforms.searchFuntions.ShapeletSearch;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.ImprovedOnlineSubSeqDistance;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.SubSeqDistance;

/**
 * A filter to transform a dataset by k shapelets. Once built on a training set,
 * the filter can be used to transform subsequent datasets using the extracted
 * shapelets.
 * <p>
 * See <a href=
 * "http://delivery.acm.org/10.1145/2340000/2339579/p289-lines.pdf?ip=139.222.14.198&acc=ACTIVE%20SERVICE&CFID=221649628&CFTOKEN=31860141&__acm__=1354814450_3dacfa9c5af84445ea2bfd7cc48180c8"
 * >Lines, J., Davis, L., Hills, J., Bagnall, A.: A shapelet transform for time
 * series classification. In: Proc. 18th ACM SIGKDD (2012)</a>
 *
 * @author Aaron Bostrom
 */
public class ShapeletTransformBasedOnLFDP extends SimpleBatchFilter {

	// Variables for experiments
	protected static long subseqDistOpCount;

	@Override
	public String globalInfo() {
		throw new UnsupportedOperationException("Not supported yet.");
	}

	// this int is used to serliase our position when iterating through a
	// dataset.
	public int dataSet;

	protected boolean supressOutput; // defaults to print in System.out AS WELL
										// as file, set to true to stop printing
										// to console
	protected int numShapelets;
	protected boolean shapeletsTrained;
	protected ArrayList<Shapelet> shapelets;
	protected String ouputFileLocation = "defaultShapeletOutput.txt"; // default
																		// store
																		// location
	protected boolean recordShapelets; // default action is to write an output
										// file
	protected boolean roundRobin;

	public final static int DEFAULT_NUMSHAPELETS = 100;
	public final static int DEFAULT_MINSHAPELETLENGTH = 3;
	public final static int DEFAULT_MAXSHAPELETLENGTH = 23;

	protected transient QualityMeasures.ShapeletQualityMeasure qualityMeasure;
	protected transient QualityMeasures.ShapeletQualityChoice qualityChoice;
	protected transient QualityBound.ShapeletQualityBound qualityBound;

	protected boolean useCandidatePruning;
	protected boolean useRoundRobin;

	protected Comparator shapeletComparator;

	protected SubSeqDistance subseqDistance;
	protected NormalClassValue classValue;
	protected ShapeletSearch searchFunction;
	protected String serialName;
	protected Shapelet worstShapelet;

	protected Instances inputData;

	protected ArrayList<Shapelet> kShapelets;

	protected long count;
	
	public void setSubSeqDistance(SubSeqDistance ssd) {
		subseqDistance = ssd;
	}

	public long getCount() {
		return count;
	}

	public void setClassValue(NormalClassValue cv) {
		classValue = cv;
	}

	public void setSearchFunction(ShapeletSearch shapeletSearch) {
		searchFunction = shapeletSearch;
	}

	public void setSerialName(String sName) {
		serialName = sName;
	}

	public void useSeparationGap() {
		shapeletComparator = new Shapelet.ReverseSeparationGap();
	}

	public void setUseRoundRobin(boolean b) {
		useRoundRobin = b;
	}

	public SubSeqDistance getSubSequenceDistance() {
		return subseqDistance;
	}

	protected int candidatePruningStartPercentage;

	protected static final double ROUNDING_ERROR_CORRECTION = 0.000000000000001;
	protected int[] dataSourceIDs;

	/**
	 * Default constructor; Quality measure defaults to information gain.
	 */
	public ShapeletTransformBasedOnLFDP() {
		this(DEFAULT_NUMSHAPELETS, DEFAULT_MINSHAPELETLENGTH, DEFAULT_MAXSHAPELETLENGTH, QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
	}

	/**
	 * Constructor for generating a shapelet transform from an ArrayList of
	 * Shapelets.
	 *
	 * @param shapes
	 */
	public ShapeletTransformBasedOnLFDP(ArrayList<Shapelet> shapes) {
		this();
		this.shapelets = shapes;
		this.shapeletsTrained = true;
		this.numShapelets = shapelets.size();
	}

	/**
	 * Single param constructor: Quality measure defaults to information gain.
	 *
	 * @param k
	 *            the number of shapelets to be generated
	 */
	public ShapeletTransformBasedOnLFDP(int k) {
		this(k, DEFAULT_MINSHAPELETLENGTH, DEFAULT_MAXSHAPELETLENGTH, QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
	}

	/**
	 * Full constructor to create a usable filter. Quality measure defaults to
	 * information gain.
	 *
	 * @param k
	 *            the number of shapelets to be generated
	 * @param minShapeletLength
	 *            minimum length of shapelets
	 * @param maxShapeletLength
	 *            maximum length of shapelets
	 */
	public ShapeletTransformBasedOnLFDP(int k, int minShapeletLength, int maxShapeletLength) {
		this(k, minShapeletLength, maxShapeletLength, QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

	}

	/**
	 * Full, exhaustive, constructor for a filter. Quality measure set via enum,
	 * invalid selection defaults to information gain.
	 *
	 * @param k
	 *            the number of shapelets to be generated
	 * @param minShapeletLength
	 *            minimum length of shapelets
	 * @param maxShapeletLength
	 *            maximum length of shapelets
	 * @param qualityChoice
	 *            the shapelet quality measure to be used with this filter
	 */
	public ShapeletTransformBasedOnLFDP(int k, int minShapeletLength, int maxShapeletLength, weka.core.shapelet.QualityMeasures.ShapeletQualityChoice qualityChoice) {
		this.numShapelets = k;
		this.shapelets = new ArrayList<>();
		this.shapeletsTrained = false;
		this.useCandidatePruning = false;
		this.qualityChoice = qualityChoice;
		this.supressOutput = false;
		this.dataSet = 0;
		this.recordShapelets = true; // default action is to write an output
										// file
		this.roundRobin = false;
		this.useRoundRobin = false;
		this.shapeletComparator = new Shapelet.ReverseOrder();
		this.kShapelets = new ArrayList<>();

		setQualityMeasure(qualityChoice);
		this.subseqDistance = new SubSeqDistance();
		this.classValue = new NormalClassValue();
		this.searchFunction = new ShapeletSearch(1, 1);
	}

	/**
	 * Returns the set of shapelets for this transform as an ArrayList.
	 *
	 * @return An ArrayList of Shapelets representing the shapelets found for
	 *         this Shapelet Transform.
	 */
	public ArrayList<Shapelet> getShapelets() {
		return this.shapelets;
	}

	/**
	 * Set the transform to round robin the data or not. This transform defaults
	 * round robin to false to keep the instances in the same order as the
	 * original data. If round robin is set to true, the transformed data will
	 * be reordered which can make it more difficult to use the ensemble.
	 *
	 * @param val
	 */
	public void setRoundRobin(boolean val) {
		this.roundRobin = val;
	}

	/**
	 * Supresses filter output to the console; useful when running timing
	 * experiments.
	 */
	public void supressOutput() {
		this.supressOutput = true;
	}

	/**
	 * Use candidate pruning technique when checking candidate quality. This
	 * speeds up the transform processing time.
	 */
	public void useCandidatePruning() {
		this.useCandidatePruning = true;
		this.candidatePruningStartPercentage = 10;
	}

	/**
	 * Use candidate pruning technique when checking candidate quality. This
	 * speeds up the transform processing time.
	 *
	 * @param percentage
	 *            the percentage of data to be precocessed before pruning is
	 *            initiated. In most cases the higher the percentage the less
	 *            effective pruning becomes
	 */
	public void useCandidatePruning(int percentage) {
		this.useCandidatePruning = true;
		this.candidatePruningStartPercentage = percentage;
	}

	/**
	 * Mutator method to set the number of shapelets to be stored by the filter.
	 *
	 * @param k
	 *            the number of shapelets to be generated
	 */
	public void setNumberOfShapelets(int k) {
		this.numShapelets = k;
	}

	/**
	 *
	 * @return
	 */
	public int getNumberOfShapelets() {
		return numShapelets;
	}

	/**
	 * Turns off log saving; useful for timing experiments where speed is
	 * essential.
	 */
	public void turnOffLog() {
		this.recordShapelets = false;
	}

	/**
	 * Set file path for the filter log. Filter log includes shapelet quality,
	 * seriesId, startPosition, and content for each shapelet.
	 *
	 * @param fileName
	 *            the updated file path of the filter log
	 */
	public void setLogOutputFile(String fileName) {
		this.recordShapelets = true;
		this.ouputFileLocation = fileName;
	}

	/**
	 *
	 * @return
	 */
	public boolean foundShapelets() {
		return shapeletsTrained;
	}

	/**
	 * Mutator method to set the minimum and maximum shapelet lengths for the
	 * filter.
	 *
	 * @param minShapeletLength
	 *            minimum length of shapelets
	 * @param maxShapeletLength
	 *            maximum length of shapelets
	 */
	// public void setShapeletMinAndMax(int min, int max) {
	// searchFunction.setMinAndMax(min, max);
	// }

	/**
	 * Mutator method to set the quality measure used by the filter. As with
	 * constructors, default selection is information gain unless another valid
	 * selection is specified.
	 *
	 * @return
	 */
	public QualityMeasures.ShapeletQualityChoice getQualityMeasure() {
		return qualityChoice;
	}

	/**
	 *
	 * @param qualityChoice
	 */
	public void setQualityMeasure(QualityMeasures.ShapeletQualityChoice qualityChoice) {
		this.qualityChoice = qualityChoice;
		switch (qualityChoice) {
		case F_STAT:
			this.qualityMeasure = new QualityMeasures.FStat();
			break;
		case KRUSKALL_WALLIS:
			this.qualityMeasure = new QualityMeasures.KruskalWallis();
			break;
		case MOODS_MEDIAN:
			this.qualityMeasure = new QualityMeasures.MoodsMedian();
			break;
		default:
			this.qualityMeasure = new QualityMeasures.InformationGain();
		}
	}

	/**
	 *
	 * @param classDist
	 * @return
	 */
	protected void initQualityBound(ClassDistribution classDist) {
		if (!useCandidatePruning)
			return;

		switch (qualityChoice) {
		case F_STAT:
			this.qualityBound = new QualityBound.FStatBound(classDist, candidatePruningStartPercentage);
			break;
		case KRUSKALL_WALLIS:
			this.qualityBound = new QualityBound.KruskalWallisBound(classDist, candidatePruningStartPercentage);
			break;
		case MOODS_MEDIAN:
			this.qualityBound = new QualityBound.MoodsMedianBound(classDist, candidatePruningStartPercentage);
			break;
		default:
			this.qualityBound = new QualityBound.InformationGainBound(classDist, candidatePruningStartPercentage);
		}
	}

	/**
	 *
	 * @param f
	 */
	public void setCandidatePruning(boolean f) {
		this.useCandidatePruning = f;
		this.candidatePruningStartPercentage = f ? 10 : 100;
	}

	/**
	 * Sets the format of the filtered instances that are output. I.e. will
	 * include k attributes each shapelet distance and a class value
	 *
	 * @param inputFormat
	 *            the format of the input data
	 * @return a new Instances object in the desired output format
	 */
	// TODO: Fix depecrated FastVector
	@Override
	protected Instances determineOutputFormat(Instances inputFormat) throws IllegalArgumentException {

		if (this.numShapelets < 1) {
			throw new IllegalArgumentException("ShapeletFilter not initialised correctly - please specify a value of k that is greater than or equal to 1");
		}

		// Set up instances size and format.
		// int length = this.numShapelets;
		int length = this.shapelets.size();
		FastVector atts = new FastVector();
		String name;
		for (int i = 0; i < length; i++) {
			name = "Shapelet_" + i;
			atts.addElement(new Attribute(name));
		}

		if (inputFormat.classIndex() >= 0) {
			// Classification set, set class
			// Get the class values as a fast vector
			Attribute target = inputFormat.attribute(inputFormat.classIndex());

			FastVector vals = new FastVector(target.numValues());
			for (int i = 0; i < target.numValues(); i++) {
				vals.addElement(target.value(i));
			}
			atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
		}
		Instances result = new Instances("Shapelets" + inputFormat.relationName(), atts, inputFormat.numInstances());
		if (inputFormat.classIndex() >= 0) {
			result.setClassIndex(result.numAttributes() - 1);
		}
		return result;
	}

	protected void inputCheck(Instances dataInst) throws IllegalArgumentException {
		if (numShapelets < 1) {
			throw new IllegalArgumentException("Number of shapelets initialised incorrectly - please select value of k (Usage: setNumberOfShapelets");
		}

		int maxPossibleLength;
		maxPossibleLength = dataInst.instance(0).numAttributes();

		if (dataInst.classIndex() >= 0) {
			maxPossibleLength -= 1;
		}

	}

	/**
	 * The main logic of the filter; when called for the first time, k shapelets
	 * are extracted from the input Instances 'data'. The input 'data' is
	 * transformed by the k shapelets, and the filtered data is returned as an
	 * output.
	 * <p>
	 * If called multiple times, shapelet extraction DOES NOT take place again;
	 * once k shapelets are established from the initial call to process(), the
	 * k shapelets are used to transform subsequent Instances.
	 * <p>
	 * Intended use:
	 * <p>
	 * 1. Extract k shapelets from raw training data to build filter;
	 * <p>
	 * 2. Use the filter to transform the raw training data into transformed
	 * training data;
	 * <p>
	 * 3. Use the filter to transform the raw testing data into transformed
	 * testing data (e.g. filter never extracts shapelets from training data,
	 * therefore avoiding bias);
	 * <p>
	 * 4. Build a classifier using transformed training data, perform
	 * classification on transformed test data.
	 *
	 * @param data
	 *            the input data to be transformed (and to find the shapelets if
	 *            this is the first run)
	 * @return the transformed representation of data, according to the
	 *         distances from each instance to each of the k shapelets
	 */
	@Override
	public Instances process(Instances data) throws IllegalArgumentException {
		inputData = data;

		// check the input data is correct and assess whether the filter has
		// been setup correctly.
		inputCheck(data);

		// setup classsValue
		classValue.init(data);

		// setup subseqDistance
		subseqDistance.init(data);

		// checks if the shapelets haven't been found yet, finds them if it
		// needs too.
		if (!shapeletsTrained) {
			trainShapelets(data);
		}

		// we log the count from the subseqdistance before we reset it in the
		// transform.
		count = subseqDistance.getCount();

		// build the transformed dataset with the shapelets we've found either
		// on this data, or the previous training data
		return buildTansformedDataset(data, shapelets);
	}

	protected void trainShapelets(Instances data) {
		// we might round robin the data in here. So we return the changed
		// dataset.
		Instances dataset = initDataSouce(data);
		shapelets = findBestKShapeletsCacheBasedOnLFDP(dataset); // get k
																// shapelets
		shapeletsTrained = true;

		// outputPrint(shapelets.size() + " Shapelets have been generated");

		// we don't need to undo the roundRobin because we clone the data into a
		// different order.
	}

	private Instances initDataSouce(Instances data) {

		int dataSize = data.numInstances();
		// shapelets discovery has not yet been caried out, so this must be
		// training data
		dataSourceIDs = new int[dataSize];

		Instances dataset = data;
		if (roundRobin) {
			// Reorder the data in round robin order
			dataset = roundRobinData(data, dataSourceIDs);
		} else {
			for (int i = 0; i < dataSize; i++) {
				dataSourceIDs[i] = i;
			}
		}

		return dataset;
	}

	protected Instances buildTansformedDataset(Instances data, ArrayList<Shapelet> shapelets) {
		// Reorder the training data and reset the shapelet indexes
		Instances output = determineOutputFormat(data);

		// reinit our data.
		subseqDistance.init(data);

		Shapelet s;
		// for each data, get distance to each shapelet and create new instance
		int size = shapelets.size();
		int dataSize = data.numInstances();

		// create our data instances
		for (int j = 0; j < dataSize; j++) {
			output.add(new DenseInstance(size + 1));
		}

		double dist;
		for (int i = 0; i < size; i++) {
			s = shapelets.get(i);
			subseqDistance.setShapelet(s);

			for (int j = 0; j < dataSize; j++) {
				dist = subseqDistance.calculate(data.instance(j).toDoubleArray(), j);
				output.instance(j).setValue(i, dist);
			}
		}

		// do the classValues.
		for (int j = 0; j < dataSize; j++) {
			// we always want to write the true ClassValue here. Irrelevant of
			// binarised or not.
			output.instance(j).setValue(size, classValue.getUnAlteredClassValue(data.instance(j)));
		}
		return output;
	}

	public ArrayList<Shapelet> findBestKShapeletsCacheBasedOnLFDP(Instances data) {
		ArrayList<Shapelet> seriesShapelets; // temp store of all shapelets for
												// each time series

		// for all time series
		// outputPrint("Processing data: ");

		int dataSize = data.numInstances();

		// for all possible time series.
		for (; dataSet < dataSize; dataSet++) {
			// outputPrint("data : " + dataSet);
			//System.out.println(dataSet+"\t"+new Date()); 
			// set the worst Shapelet so far, as long as the shapelet set is
			// full.
			worstShapelet = kShapelets.size() == numShapelets ? kShapelets.get(numShapelets - 1) : null;

			// set the series we're working with.
			subseqDistance.setSeries(dataSet);
			// set the clas value of the series we're working with.
			classValue.setShapeletValue(data.get(dataSet));

			seriesShapelets = searchFunction.SearchForShapeletsInSeriesBasedLFDP(data.get(dataSet), this::checkCandidate);

			Collections.sort(seriesShapelets, shapeletComparator);

			seriesShapelets = removeSelfSimilar(seriesShapelets);

			kShapelets = combine(numShapelets, kShapelets, seriesShapelets);

			createSerialFile();
		}

		this.numShapelets = kShapelets.size();

		recordShapelets(kShapelets, this.ouputFileLocation);
		// printShapelets(kShapelets);

		return kShapelets;
	}

	public void createSerialFile() {
		if (serialName == null)
			return;

		// Serialise the object.
		ObjectOutputStream out = null;
		try {
			out = new ObjectOutputStream(new FileOutputStream(serialName));
			out.writeObject(this);
		} catch (IOException ex) {
			System.out.println("Failed to write " + ex);
		} finally {
			if (out != null) {
				try {
					out.close();
				} catch (IOException ex) {
					System.out.println("Failed to close " + ex);
				}
			}
		}
	}

	public ArrayList<Shapelet> findBestKShapeletsCacheBasedOnLFDP(int numShapelets, Instances data) {
		this.numShapelets = numShapelets;
		// setup classsValue
		classValue.init(data);
		// setup subseqDistance
		subseqDistance.init(data);
		initDataSouce(data);
		return findBestKShapeletsCacheBasedOnLFDP(data);
	}

	/**
	 * Private method to combine two ArrayList collections of
	 * FullShapeletTransform objects.
	 *
	 * @param k
	 *            the maximum number of shapelets to be returned after combining
	 *            the two lists
	 * @param kBestSoFar
	 *            the (up to) k best shapelets that have been observed so far,
	 *            passed in to combine with shapelets from a new series (sorted)
	 * @param timeSeriesShapelets
	 *            the shapelets taken from a new series that are to be merged in
	 *            descending order of fitness with the kBestSoFar
	 * @return an ordered ArrayList of the best k (or less) (sorted)
	 *         FullShapeletTransform objects from the union of the input
	 *         ArrayLists
	 */
	protected ArrayList<Shapelet> combine(int k, ArrayList<Shapelet> kBestSoFar, ArrayList<Shapelet> timeSeriesShapelets) {
		// both kBestSofar and timeSeries are sorted so we can explot this.
		// maintain a pointer for each list.
		ArrayList<Shapelet> newBestSoFar = new ArrayList<>();

		// best so far pointer
		int bsfPtr = 0;
		// new time seris pointer.
		int tssPtr = 0;

		for (int i = 0; i < k; i++) {
			Shapelet shapelet1 = null, shapelet2 = null;

			if (bsfPtr < kBestSoFar.size()) {
				shapelet1 = kBestSoFar.get(bsfPtr);
			}
			if (tssPtr < timeSeriesShapelets.size()) {
				shapelet2 = timeSeriesShapelets.get(tssPtr);
			}

			boolean shapelet1Null = shapelet1 == null;
			boolean shapelet2Null = shapelet2 == null;

			// both lists have been explored, but we have less than K elements.
			if (shapelet1Null && shapelet2Null) {
				break;
			}

			// one list is expired keep adding the other list until we reach K.
			if (shapelet1Null) {
				newBestSoFar.add(shapelet2);
				tssPtr++;
				continue;
			}

			// one list is expired keep adding the other list until we reach K.
			if (shapelet2Null) {
				newBestSoFar.add(shapelet1);
				bsfPtr++;
				continue;
			}

			// if both lists are fine then we need to compare which one to use.
			if (shapeletComparator.compare(shapelet1, shapelet2) == -1) {
				newBestSoFar.add(shapelet1);
				bsfPtr++;
				shapelet1 = null;
			} else {
				newBestSoFar.add(shapelet2);
				tssPtr++;
				shapelet2 = null;
			}

		}

		return newBestSoFar;
	}

	/**
	 * protected method to remove self-similar shapelets from an ArrayList (i.e.
	 * if they come from the same series and have overlapping indicies)
	 *
	 * @param shapelets
	 *            the input Shapelets to remove self similar
	 *            FullShapeletTransform objects from
	 * @return a copy of the input ArrayList with self-similar shapelets removed
	 */
	protected static ArrayList<Shapelet> removeSelfSimilar(ArrayList<Shapelet> shapelets) {
		// return a new pruned array list - more efficient than removing
		// self-similar entries on the fly and constantly reindexing
		ArrayList<Shapelet> outputShapelets = new ArrayList<>();
		int size = shapelets.size();
		boolean[] selfSimilar = new boolean[size];

		for (int i = 0; i < size; i++) {
			if (selfSimilar[i]) {
				continue;
			}

			outputShapelets.add(shapelets.get(i));

			for (int j = i + 1; j < size; j++) {
				// no point recalc'ing if already self similar to something
				if ((!selfSimilar[j]) && selfSimilarity(shapelets.get(i), shapelets.get(j))) {
					selfSimilar[j] = true;
				}
			}
		}
		return outputShapelets;
	}

	protected Shapelet checkCandidate(double[] series, int start, int length) {
		// init qualityBound.

		initQualityBound(classValue.getClassDistributions());

		// Set bound of the bounding algorithm
		if (qualityBound != null && worstShapelet != null) {
			qualityBound.setBsfQuality(worstShapelet.qualityValue);
		}

		double[] candidate = new double[length];

		// copy the data from the whole series into a candidate.
		System.arraycopy(series, start, candidate, 0, length);

		// znorm candidate here so it's only done once, rather than in each
		// distance calculation
		candidate = subseqDistance.zNormalise(candidate, false);

		// create orderline by looping through data set and calculating the
		// subsequence
		// distance from candidate to all data, inserting in order.
		ArrayList<OrderLineObj> orderline = new ArrayList<>();

		int dataSize = inputData.numInstances();

		subseqDistance.setCandidate(candidate, start);

		for (int i = 0; i < dataSize; i++) {
			// Check if it is possible to prune the candidate
			if (qualityBound != null && qualityBound.pruneCandidate()) {
				return null;
			}

			double distance = 0.0;
			// don't compare the shapelet to the the time series it came from
			// because we know it's 0.
			if (i != dataSet) {
				distance = subseqDistance.calculate(inputData.instance(i).toDoubleArray(), i);
			}

			// this could be binarised or normal.
			double classVal = classValue.getClassValue(inputData.instance(i));

			// without early abandon, it is faster to just add and sort at the
			// end
			orderline.add(new OrderLineObj(distance, classVal));

			// Update qualityBound - presumably each bounding method for
			// different quality measures will have a different update
			// procedure.
			if (qualityBound != null) {
				qualityBound.updateOrderLine(orderline.get(orderline.size() - 1));
			}
		}

		Shapelet shapelet = new Shapelet(candidate, dataSourceIDs[dataSet], start, this.qualityMeasure);
		// this class distribution could be binarised or normal.
		shapelet.calculateQuality(orderline, classValue.getClassDistributions());
		shapelet.classValue = classValue.getShapeletValue(); // set classValue
																// of shapelet.
																// (interesing
																// to know).
		return shapelet;
	}

	/**
	 * Load a set of Instances from an ARFF
	 *
	 * @param fileName
	 *            the file name of the ARFF
	 * @return a set of Instances from the ARFF
	 */
	public static Instances loadData(String fileName) {
		Instances data = null;
		try {
			FileReader r;
			r = new FileReader(fileName);
			data = new Instances(r);

			data.setClassIndex(data.numAttributes() - 1);
		} catch (IOException e) {
			System.out.println(" Error =" + e + " in method loadData");
		}
		return data;
	}

	/**
	 * A private method to assess the self similarity of two
	 * FullShapeletTransform objects (i.e. whether they have overlapping
	 * indicies and are taken from the same time series).
	 *
	 * @param shapelet
	 *            the first FullShapeletTransform object (in practice, this will
	 *            be the dominant shapelet with quality >= candidate)
	 * @param candidate
	 *            the second FullShapeletTransform
	 * @return
	 */
	private static boolean selfSimilarity(Shapelet shapelet, Shapelet candidate) {
		if (candidate.seriesId == shapelet.seriesId) {
			if (candidate.startPos >= shapelet.startPos && candidate.startPos < shapelet.startPos + shapelet.content.length) { // exisiting
																																// //
																																// shapelet
				return true;
			}
			if (shapelet.startPos >= candidate.startPos && shapelet.startPos < candidate.startPos + candidate.content.length) {
				return true;
			}
		}
		return false;
	}

	/**
	 * A method to read in a FullShapeletTransform log file to reproduce a
	 * FullShapeletTransform
	 * <p>
	 * NOTE: assumes shapelets from log are Z-NORMALISED
	 *
	 * @param fileName
	 *            the name and path of the log file
	 * @return a duplicate FullShapeletTransform to the object that created the
	 *         original log file
	 * @throws Exception
	 */
	public static ShapeletTransformBasedOnLFDP createFilterFromFile(String fileName) throws Exception {
		return createFilterFromFile(fileName, Integer.MAX_VALUE);
	}

	public double timingForSingleShapeletBasedOnLFDP(Instances data) {
		data = roundRobinData(data, null);
		long startTime = System.nanoTime();
		findBestKShapeletsCacheBasedOnLFDP(1, data);
		long finishTime = System.nanoTime();
		return (double) (finishTime - startTime) / 1000000000.0;
	}

	protected void recordShapelets(ArrayList<Shapelet> kShapelets, String saveLocation) {
		if (!this.recordShapelets) {
			return;
		}

		try {
			// just in case the file doesn't exist or the directories.
			File file = new File(saveLocation);
			if (file.getParentFile() != null) {
				file.getParentFile().mkdirs();
			}

			FileWriter out = new FileWriter(file);

			for (Shapelet kShapelet : kShapelets) {
				out.append(kShapelet.qualityValue + "," + kShapelet.seriesId + "," + kShapelet.startPos + "\n");
				double[] shapeletContent = kShapelet.content;

				for (int j = 0; j < shapeletContent.length; j++) {
					out.append(shapeletContent[j] + ",");
				}
				out.append("\n");
			}
			out.close();
		} catch (IOException ex) {
			System.out.println("IOException: " + ex);
		}

	}

	protected void printShapelets(ArrayList<Shapelet> kShapelets) {
		if (supressOutput) {
			return;
		}

		System.out.println();
		System.out.println("Output Shapelets:");
		System.out.println("-------------------");
		System.out.println("informationGain,seriesId,startPos,classVal");
		System.out.println("<shapelet>");
		System.out.println("-------------------");
		System.out.println();
		for (Shapelet kShapelet : kShapelets) {
			System.out.println(kShapelet.qualityValue + "," + kShapelet.seriesId + "," + kShapelet.startPos + "," + kShapelet.classValue);
			double[] shapeletContent = kShapelet.content;
			for (int j = 0; j < shapeletContent.length; j++) {
				System.out.print(shapeletContent[j] + ",");
			}
			System.out.println();
		}

	}

	/**
	 * Returns a list of the lengths of the shapelets found by this transform.
	 *
	 * @return An ArrayList of Integers representing the lengths of the
	 *         shapelets.
	 */
	public ArrayList<Integer> getShapeletLengths() {
		ArrayList<Integer> shapeletLengths = new ArrayList<>();

		if (this.shapeletsTrained) {
			for (Shapelet s : this.shapelets) {
				shapeletLengths.add(s.content.length);
			}
		}

		return shapeletLengths;
	}

	/**
	 * A method to read in a FullShapeletTransform log file to reproduce a
	 * FullShapeletTransform,
	 * <p>
	 * NOTE: assumes shapelets from log are Z-NORMALISED
	 *
	 * @param fileName
	 *            the name and path of the log file
	 * @param maxShapelets
	 * @return a duplicate FullShapeletTransform to the object that created the
	 *         original log file
	 * @throws Exception
	 */
	public static ShapeletTransformBasedOnLFDP createFilterFromFile(String fileName, int maxShapelets) throws Exception {

		File input = new File(fileName);
		Scanner scan = new Scanner(input);
		scan.useDelimiter("\n");

		ShapeletTransformBasedOnLFDP sf = new ShapeletTransformBasedOnLFDP();
		ArrayList<Shapelet> shapelets = new ArrayList<>();

		String shapeletContentString;
		String shapeletStatsString;
		ArrayList<Double> content;
		double[] contentArray;
		Scanner lineScan;
		Scanner statScan;
		double qualVal;
		int serID;
		int starPos;

		int shapeletCount = 0;

		while (shapeletCount < maxShapelets && scan.hasNext()) {
			shapeletStatsString = scan.next();
			shapeletContentString = scan.next();

			// Get the shapelet stats
			statScan = new Scanner(shapeletStatsString);
			statScan.useDelimiter(",");

			qualVal = Double.parseDouble(statScan.next().trim());
			serID = Integer.parseInt(statScan.next().trim());
			starPos = Integer.parseInt(statScan.next().trim());
			// End of shapelet stats

			lineScan = new Scanner(shapeletContentString);
			lineScan.useDelimiter(",");

			content = new ArrayList<>();
			while (lineScan.hasNext()) {
				String next = lineScan.next().trim();
				if (!next.isEmpty()) {
					content.add(Double.parseDouble(next));
				}
			}

			contentArray = new double[content.size()];
			for (int i = 0; i < content.size(); i++) {
				contentArray[i] = content.get(i);
			}

			contentArray = sf.subseqDistance.zNormalise(contentArray, false);

			Shapelet s = new Shapelet(contentArray, qualVal, serID, starPos);

			shapelets.add(s);
			shapeletCount++;
		}
		sf.shapelets = shapelets;
		sf.shapeletsTrained = true;
		sf.numShapelets = shapelets.size();

		return sf;
	}

	/**
	 * Method to reorder the given Instances in round robin order
	 *
	 * @param data
	 *            Instances to be reordered
	 * @param sourcePos
	 *            Pointer to array of ints, where old positions of instances are
	 *            to be stored.
	 * @return Instances in round robin order
	 */
	public static Instances roundRobinData(Instances data, int[] sourcePos) {
		// Count number of classes
		TreeMap<Double, ArrayList<Instance>> instancesByClass = new TreeMap<>();
		TreeMap<Double, ArrayList<Integer>> positionsByClass = new TreeMap<>();

		NormalClassValue ncv = new NormalClassValue();
		ncv.init(data);

		// Get class distributions
		ClassDistribution classDistribution = ncv.getClassDistributions();

		// Allocate arrays for instances of every class
		for (int i = 0; i < classDistribution.size(); i++) {
			int frequency = classDistribution.get(i);
			instancesByClass.put((double) i, new ArrayList<>(frequency));
			positionsByClass.put((double) i, new ArrayList<>(frequency));
		}

		int dataSize = data.numInstances();
		// Split data according to their class memebership
		for (int i = 0; i < dataSize; i++) {
			Instance inst = data.instance(i);
			instancesByClass.get(ncv.getClassValue(inst)).add(inst);
			positionsByClass.get(ncv.getClassValue(inst)).add(i);
		}

		// Merge data into single list in round robin order
		Instances roundRobinData = new Instances(data, dataSize);
		for (int i = 0; i < dataSize;) {
			// Allocate arrays for instances of every class
			for (int j = 0; j < classDistribution.size(); j++) {
				ArrayList<Instance> currentList = instancesByClass.get((double) j);
				ArrayList<Integer> currentPositions = positionsByClass.get((double) j);

				if (!currentList.isEmpty()) {
					roundRobinData.add(currentList.remove(currentList.size() - 1));
					if (sourcePos != null && sourcePos.length == dataSize) {
						sourcePos[i] = currentPositions.remove(currentPositions.size() - 1);
					}
					i++;
				}
			}
		}

		return roundRobinData;
	}

	public void outputPrint(String val) {
		if (!this.supressOutput) {
			System.out.println(val);
		}
	}

	@Override
	public String toString() {
		String str = "Shapelets: ";
		for (Shapelet s : shapelets) {
			str += s.toString() + "\n";
		}
		return str;
	}

	/**
	 *
	 * @param data
	 * @param minShapeletLength
	 * @param maxShapeletLength
	 * @return
	 * @throws Exception
	 */
	public long opCountForSingleShapeletBasedOnLFDP(Instances data) throws Exception {
		data = roundRobinData(data, null);
		subseqDistOpCount = 0;
		findBestKShapeletsCacheBasedOnLFDP(1, data);
		return subseqDistOpCount;
	}

	public static void main(String[] args) {
		//String[] problems = {"ChlorineConcentration", "Coffee", "ECGFiveDays", "GunPoint", "Lightning7", "MedicalImages", "MoteStrain", "Symbols", "Trace", "TwoLeadECG" };
		//String[] problems = {"Coffee", "ECGFiveDays", "GunPoint", "Lightning7", "MedicalImages", "MoteStrain", "Symbols", "Trace", "TwoLeadECG" };
		String[] problems ={"conductance"};
		System.out.println(problems.length);
		// System.out.println("dataset\t" + "C45\t"+"1NN\t" + "BN\t" +
		// "bayesNet\t" + "RandF\t" + "RotF\t"+ "SVM\t" + "WeightedEnsemble");
		for (int i = 0; i < problems.length; i++) {
			System.out.print(problems[i] + "\t");
			trainTestExample(problems[i]);
			System.out.println();
		}

	}

	public static void trainTestExample(String problem) {
		try {
			final String resampleLocation = DataSets.problemPath;
			final String dataset = problem;
			final String filePath = resampleLocation + File.separator + dataset + File.separator + dataset;
			Instances test, train;
			test = utilities.ClassifierTools.loadData(filePath + "_TEST");
			train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
			// use fold as the seed.
			// train = InstanceTools.subSample(train, 100, fold);
			ShapeletTransformBasedOnLFDP transform = new ShapeletTransformBasedOnLFDP();
			transform.setRoundRobin(true);
			// construct shapelet classifiers.
			transform.setClassValue(new BinarisedClassValue());
			transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
			transform.useCandidatePruning();
			transform.setNumberOfShapelets(train.numInstances() / 2);
			transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
			long d1 = System.nanoTime();
			Instances tranTrain = transform.process(train);
			Instances tranTest = transform.process(test);
			long d2 = System.nanoTime();
			ArrayList<Shapelet> sh = transform.getShapelets();
			System.out.print((d2 - d1) * 0.000000001+ "\t");

			 //1C45
			 J48 tree = new J48();
			 tree.buildClassifier(tranTrain);
			 double accuracy = ClassifierTools.accuracy(tranTest, tree);
			 System.out.print(accuracy + "\t");
			
			 // 2 1NN
			 kNN k = new kNN(1);
			 k.buildClassifier(tranTrain);
			 accuracy = ClassifierTools.accuracy(tranTest, k);
			 System.out.print(accuracy + "\t");
			
			 // 3BN
			 NaiveBayes nb = new NaiveBayes();
			 nb.buildClassifier(tranTrain);
			 accuracy = ClassifierTools.accuracy(tranTest, nb);
			 System.out.print(accuracy + "\t");
			
			 // 4bayesNet
			 BayesNet bn = new BayesNet();
			 bn.buildClassifier(tranTrain);
			 accuracy = ClassifierTools.accuracy(tranTest, bn);
			 System.out.print(accuracy + "\t");
			
			 // 5RandF
			 EnhancedRandomForest erf = new EnhancedRandomForest();
			 erf.setNumTrees(500);
			 erf.buildClassifier(tranTrain);
			 accuracy = ClassifierTools.accuracy(tranTest, erf);
			 System.out.print(accuracy + "\t");
			
			 // 6RotF
			 RotationForest rf = new RotationForest();
			 rf.setNumIterations(50);
			 rf.buildClassifier(tranTrain);
			 accuracy = ClassifierTools.accuracy(tranTest, rf);
			 System.out.print(accuracy + "\t");
			
			 // 7SVML
			 SMO svml = new SMO();
			 PolyKernel kernel = new PolyKernel();
			 kernel.setExponent(1);
			 svml.setKernel(kernel);
			 svml.buildClassifier(tranTrain);
			 accuracy = ClassifierTools.accuracy(tranTest, svml);
			 System.out.print(accuracy + "\t");
			
			 // 8WeightedEnsemble
			 WeightedEnsemble we = new WeightedEnsemble();
			 we.buildClassifier(tranTrain);
			 accuracy = ClassifierTools.accuracy(tranTest, we);
			 System.out.print(accuracy+"\t");

		} catch (Exception ex) {
			Logger.getLogger(ShapeletTransformBasedOnLFDP.class.getName()).log(Level.SEVERE, null, ex);
		}
	}


	
}
