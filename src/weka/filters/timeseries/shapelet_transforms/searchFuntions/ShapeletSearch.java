/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.filters.timeseries.shapelet_transforms.searchFuntions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Map;

import representation.plr.PLR_LFDP;
import representation.plr.PLR_EP;
import weka.core.Instance;
import weka.core.shapelet.Shapelet;

/**
 *
 * @author raj09hxu
 */
public class ShapeletSearch implements Serializable {

	public interface ProcessCandidate {
		public Shapelet process(double[] candidate, int start, int length);
	}

	protected int minShapeletLength;
	protected int maxShapeletLength;

	protected int lengthIncrement = 1;
	protected int positionIncrement = 1;

	public ShapeletSearch(int min, int max) {
		minShapeletLength = min;
		maxShapeletLength = max;
	}

	public ShapeletSearch(int min, int max, int lengthInc, int posInc) {
		this(min, max);
		lengthIncrement = lengthInc;
		positionIncrement = posInc;
	}

	public void setMinAndMax(int min, int max) {
		minShapeletLength = min;
		maxShapeletLength = max;
	}

	public ArrayList<Shapelet> SearchForShapeletsInSeries(Instance timeSeries, ProcessCandidate checkCandidate) {
		ArrayList<Shapelet> seriesShapelets = new ArrayList<>();

		double[] series = timeSeries.toDoubleArray();

		for (int length = minShapeletLength; length <= maxShapeletLength; length += lengthIncrement) {
			// for all possible starting positions of that length. -1 to remove
			// classValue
			for (int start = 0; start <= timeSeries.numAttributes() - length - 1; start += positionIncrement) {
				Shapelet shapelet = checkCandidate.process(series, start, length);

				if (shapelet != null) {
					seriesShapelets.add(shapelet);
				}
			}
		}

		return seriesShapelets;
	}

	public ArrayList<Shapelet> SearchForShapeletsInSeriesBasedLFDP(Instance timeSeries, ProcessCandidate checkCandidate) {
		ArrayList<Shapelet> seriesShapelets = new ArrayList<>();

		double[] series = timeSeries.toDoubleArray();
		double[] tempSeries = new double[series.length - 1];
		for (int i = 0; i < tempSeries.length; i++) {
			tempSeries[i] = series[i];
		}
		int[] LFDP = new PLR_LFDP(tempSeries).getLFDPIndexByNumber((int) Math.ceil(0.05*tempSeries.length)+2);
		
		//int[] LFDP = new PLR_LFDP(tempSeries).getLFDPIndexByNumber((int) Math.sqrt(series.length) + 1);		
		for (int start = 0; start < LFDP.length - 2; start++) {
			for (int end = start + 2; end < LFDP.length; end++) {
				//System.out.println(start+"\t"+end);
				//Shapelet shapelet = checkCandidate.process(series, LFDP[start], LFDP[end] - LFDP[start]);
				Shapelet shapelet = checkCandidate.process(series, LFDP[start], LFDP[end] - LFDP[start]);
				if (shapelet != null) {
					seriesShapelets.add(shapelet);
				}
			}
		}

		return seriesShapelets;
	}
	
	public ArrayList<Shapelet> SearchForShapeletsInSeriesBasedLFDP(Instance timeSeries, ProcessCandidate checkCandidate, double LFDPrate) {
		ArrayList<Shapelet> seriesShapelets = new ArrayList<>();

		double[] series = timeSeries.toDoubleArray();
		double[] tempSeries = new double[series.length - 1];
		for (int i = 0; i < tempSeries.length; i++) {
			tempSeries[i] = series[i];
		}
		int[] LFDP = new PLR_LFDP(tempSeries).getLFDPIndexByNumber((int) Math.ceil(LFDPrate*tempSeries.length)+2);
		
		//int[] LFDP = new PLR_LFDP(tempSeries).getLFDPIndexByNumber((int) Math.sqrt(series.length) + 1);		
		for (int start = 0; start < LFDP.length - 2; start++) {
			for (int end = start + 2; end < LFDP.length; end++) {
				//Shapelet shapelet = checkCandidate.process(series, LFDP[start], LFDP[end] - LFDP[start]);
				Shapelet shapelet = checkCandidate.process(series, LFDP[start], LFDP[end] - LFDP[start]);
				if (shapelet != null) {
					seriesShapelets.add(shapelet);
				}
			}
		}

		return seriesShapelets;
	}
	/**
	 * add by jc to find shapelet by IP
	 * 
	 * @param timeSeries
	 * @param checkCandidate
	 * @return
	 */
	public ArrayList<Shapelet> SearchForShapeletsInSeriesBasedIP(Instance timeSeries, ProcessCandidate checkCandidate) {
		ArrayList<Shapelet> seriesShapelets = new ArrayList<>();

		double[] series = timeSeries.toDoubleArray();
		double[] tempSeries = new double[series.length - 1];
		for (int i = 0; i < tempSeries.length; i++) {
			tempSeries[i] = series[i];
		}
		int[] IP = new PLR_EP().choosePointIPByNum(tempSeries);
		for (int start = 0; start < IP.length - 1; start++) {
			for (int end = start + 1; end < IP.length; end++) {
				Shapelet shapelet = checkCandidate.process(series, IP[start], IP[end] - IP[start]);
				if (shapelet != null) {
					seriesShapelets.add(shapelet);
				}
			}
		}

		return seriesShapelets;
	}

}
