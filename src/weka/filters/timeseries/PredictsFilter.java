/*
 * copyright: Anthony Bagnall
 * 
 * */
package weka.filters.timeseries;

import development.DataSets;
import fileIO.OutFile;
import java.io.FileReader;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import weka.classifiers.lazy.DTW_1NN;
import weka.classifiers.lazy.kNN;
import weka.core.*;
import weka.filters.SimpleBatchFilter;

public class PredictsFilter extends SimpleBatchFilter {
	private int predictsNum;
	private double[][] allPredictions;

	public PredictsFilter() {
		predictsNum = 35;
	}

	public PredictsFilter(int predictsNum) {
		this.predictsNum = predictsNum;
	}

	public PredictsFilter(int predictsNum, double[][] allPredictions) {
		this.predictsNum = predictsNum;
		this.allPredictions = allPredictions;
	}

	public int getPredictsNum() {
		return predictsNum;
	}

	public void setPredictsNum(int predictsNum) {
		this.predictsNum = predictsNum;
	}

	public double[][] getAllPredictions() {
		return allPredictions;
	}

	public void setAllPredictions(double[][] allPredictions) {
		this.allPredictions = allPredictions;
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
		Attribute a;
		FastVector fv = new FastVector();
		FastVector atts = new FastVector();

		for (int i = 0; i < predictsNum; i++) {
			// Add to attribute list
			String name = "Predict_" + i;
			atts.addElement(new Attribute(name));
		}
		// Get the class values as a fast vector
		Attribute target = inputFormat.attribute(inputFormat.classIndex());

		FastVector vals = new FastVector(target.numValues());
		for (int i = 0; i < target.numValues(); i++) {
			vals.addElement(target.value(i));
		}
		atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
		Instances result = new Instances("Predict" + inputFormat.relationName(), atts, inputFormat.numInstances());
		if (inputFormat.classIndex() >= 0) {
			result.setClassIndex(result.numAttributes() - 1);
		}

		return result;
	}

	@Override
	public String globalInfo() {
		return null;
	}

	@Override
	public Instances process(Instances instances) throws Exception {
		if(null==allPredictions)
			throw new Exception("Please set allPredictions");
		Instances result = determineOutputFormat(instances);
		Instance newInst;
		for (int j = 0; j < instances.numInstances(); j++) {
			newInst = new DenseInstance(result.numAttributes());
			for (int k = 0; k < predictsNum; k++) {
				newInst.setValue(k, allPredictions[j][k]);
			}
			newInst.setValue(result.classIndex(), instances.instance(j).classValue());
			result.add(newInst);
		}
		return result;
	}

	public String getRevision() {
		return null;
	}
}
