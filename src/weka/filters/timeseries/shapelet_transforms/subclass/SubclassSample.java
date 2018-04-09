package weka.filters.timeseries.shapelet_transforms.subclass;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import development.DataSets;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.timeseries.shapelet_transforms.subclass.dist.Dist;
import weka.filters.timeseries.shapelet_transforms.subclass.dist.DistCompare;

public class SubclassSample {
	private Map<Integer, String> classMap;	
	public SubclassSample() {
		super();
		// TODO Auto-generated constructor stub
		this.classMap=new HashMap<Integer, String>();
	}
	
	public Map<Integer, String> getClassMap() {
		return classMap;
	}

	public void setClassMap(Map<Integer, String> classMap) {
		this.classMap = classMap;
	}

	private double[] sumValue(Instances instances) {
		double[] sumValue = new double[instances.numInstances()];
		for (int i = 0; i < instances.numInstances(); i++) {
			sumValue[i] = 0;
			Instance instance = instances.get(i);
			for (int j = 0; j < instances.numAttributes() - 1; j++) {
				sumValue[i] += instance.value(j);
			}
		}
		return sumValue;
	}
	
	private Map<Integer, List<Integer>> classStatistics(Instances instances) {
		Map<Integer, List<Integer>> classStatistics = new HashMap<Integer, List<Integer>>();
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance instance = instances.get(i);
			int classValue = (int) instance.classValue();
			if (!classStatistics.containsKey(classValue)) {
				List<Integer> list = new ArrayList<Integer>();
				list.add(i);
				classStatistics.put(classValue, list);
			} else {
				List<Integer> list = classStatistics.get(classValue);
				list.add(i);
				classStatistics.put(classValue, list);
			}
		}
		return classStatistics;
	}
	
	private Map<Integer, Integer> classPivot(Map<Integer, List<Integer>> classStatistics, double[] sumValue) {
		Map<Integer, Integer> classPivot = new HashMap<Integer, Integer>();
		Iterator entries = classStatistics.entrySet().iterator();
		List<Integer> list = new ArrayList<Integer>();
		while (entries.hasNext()) {
			Map.Entry entry = (Map.Entry) entries.next();
			list = (ArrayList<Integer>) entry.getValue();
			double classSum = 0;
			for (int i = 0; i < list.size(); i++) {
				int index = list.get(i);
				classSum += sumValue[index];
			}
			double classMeanValue = classSum / list.size();
			int pivotIndex=list.get(0);
			double min=Math.abs(sumValue[pivotIndex]-classMeanValue);		
			for (int i = 1; i < list.size(); i++) {
				int index=list.get(i);
				double diff=Math.abs(sumValue[index]-classMeanValue);
				if(diff<min){
					min=diff;
					pivotIndex=index;
				}
			}
			classPivot.put((Integer) entry.getKey(), pivotIndex);
		}		
		return classPivot;
	}
	
	private void subClassSplittingForOneClass(Instances instances, int classValue,int privotIndex,List<Integer> instanceIndexList){
		List<Dist> distList=new ArrayList<Dist>();
		Instance pivot=instances.get(privotIndex);
		for(int i=0;i<instanceIndexList.size();i++){
			int index=instanceIndexList.get(i);
			Instance instance=instances.get(index);
			double dist=0;
			for(int j=0;j<instances.numAttributes()-1;j++){
				dist+=Math.pow(pivot.value(j)-instance.value(j), 2);
			}
			dist=Math.sqrt(dist);
			distList.add(new Dist(index,dist));
		}
		Collections.sort(distList, new DistCompare());
		
		double[] diff=new double[distList.size()-1];
		for(int i=0;i<distList.size()-1;i++){
			diff[i]=distList.get(i+1).getDist()-distList.get(i).getDist();
		}
		//double T=std(diff)/2;
		double T=std(diff);
		int C=1;
				
		classMap.put(distList.get(0).getInstanceIndex(), classValue+"_"+C);
		classMap.put(distList.get(1).getInstanceIndex(), classValue+"_"+C);
		for(int i=2;i<=diff.length;i++){
			if(diff[i-1]>T){
				C++;
			}
			classMap.put(distList.get(i).getInstanceIndex(), classValue+"_"+C);
		}
	} 
	
	public List<Integer> subClassSplitting(Instances instances){
		Map<Integer, List<Integer>> map = new SubclassSample().classStatistics(instances);
		double[] s = new SubclassSample().sumValue(instances);
		Map<Integer, Integer> classPivot = new SubclassSample().classPivot(map, s);
		
		classMap=new HashMap<Integer, String>();
		Iterator entries = map.entrySet().iterator();
		while (entries.hasNext()) {
			Map.Entry entry = (Map.Entry) entries.next();
			Integer classValue=(Integer)entry.getKey();
			List<Integer> instanceIndexList=(List<Integer>)entry.getValue();
			Integer privotIndex=classPivot.get(classValue);
			
			if(instanceIndexList.size()>=2){
				subClassSplittingForOneClass(instances,classValue,privotIndex,instanceIndexList);
			
			}else{
				classMap.put(instanceIndexList.get(0),classValue+"_1");
			}
		}
		List<Integer> sampleIndex=getSampleIndex(instances);
		//System.out.println(sampleIndex.size());
		return sampleIndex;
		
	}

	private List<Integer> getSampleIndex(Instances instances){
		double[] dist=new double[instances.numInstances()];
		for(int i=0;i<instances.numInstances();i++){
			dist[i]=0;
		}
		for(int i=0;i<instances.numInstances()-1;i++){
			for(int j=i+1;j<instances.numInstances();j++){
				if(classMap.get(i).equals(classMap.get(j))){
					Instance iInstance=instances.get(i);
					Instance jInstance=instances.get(j);
					double distance=0;
					for(int k=0;k<instances.numAttributes()-1;k++)
						distance+=Math.pow(iInstance.value(k)-jInstance.value(k), 2);
					distance=Math.sqrt(distance);		
					dist[i]+=distance;
					dist[j]+=distance;
				}
			}
		}
		Map<String,Integer> sampleIndex=new HashMap<String,Integer>();
		for(int i=0;i<instances.numInstances();i++){
			if(sampleIndex.containsKey(classMap.get(i))){
				int j=sampleIndex.get(classMap.get(i));
				if(dist[i]<dist[j]){
					sampleIndex.put(classMap.get(i),i);
				}
				
			}else{
				sampleIndex.put(classMap.get(i),i);
			}
		}
		
		Iterator entries = sampleIndex.entrySet().iterator();  
		List<Integer> list=new ArrayList<Integer>(); 
		while (entries.hasNext()) {  		  
		    Map.Entry entry = (Map.Entry) entries.next();  
		    Integer value = (Integer)entry.getValue();  
		    list.add(value); 	  
		} 
		return list;
	}
	
	private double mean(double[] data){
		int k=data.length;
		double sum=0;
		for(int i=0;i<k;i++){
			sum+=data[i];
		}
		return sum/k;
	}
	
	private double std(double[] data){
		double mean=mean(data);
		int k=data.length;
		double std=0;
		for(int i=0;i<k;i++){
			//std+=Math.pow(data[i], mean);
			std+=Math.pow(data[i]- mean,2);
		}
		//此处需要二次确定
		std=std/k;
		//std=std/(k-1);
		std=Math.sqrt(std);
		return std;
	}
	
	public static void main(String[] args) {
		final String resampleLocation = DataSets.problemPath;
		final String dataset = "ECG200";
		final String filePath = resampleLocation + File.separator + dataset + File.separator + dataset;
		Instances test, train;
		test = utilities.ClassifierTools.loadData(filePath + "_TEST");
		train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
		List<Integer> sampleIndex=new SubclassSample().subClassSplitting(train);
		System.out.println(sampleIndex.size());
	}
	
}
